from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from google.cloud import speech_v1p1beta1 as speech
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import os
import time
import uuid
import threading
import logging
import configparser
import hashlib
import random

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---
config = configparser.ConfigParser()
config.read('config.ini')

twilio_sid = config['TWILIO']['ACCOUNT_SID']
twilio_token = config['TWILIO']['AUTH_TOKEN']
twilio_phone_number = config['TWILIO']['PHONE_NUMBER']
recipient_number = config['TWILIO']['RECIPIENT_NUMBER']

gcp_credentials = config['GCP']['CREDENTIALS_PATH']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials

twilio_client = Client(twilio_sid, twilio_token)
groq_client = Groq(api_key=config['GROQ']['API_KEY'])
eleven_client = ElevenLabs(api_key=config['ELEVENLABS']['API_KEY'])
speech_client = speech.SpeechClient()  # Initialized before GCP creds are set

voice_id = config['ELEVENLABS']['VOICE_ID']
public_url = config['SERVER']['PUBLIC_URL']


class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deleted_keys = set()

    def __delitem__(self, key):
        self._deleted_keys.add(key)
        super().__delitem__(key)

    def __setitem__(self, key, value):
        if key in self._deleted_keys:
            self._deleted_keys.remove(key)
        super().__setitem__(key, value)


conversation_history = Dict()
audio_streams = {}
processing_status = {}

# --- Voice Settings ---
VOICE_SETTINGS = VoiceSettings(
    stability=0.25,
    similarity_boost=0.95,
    style=0.99,
    use_speaker_boost=True,
    optimizer="eleven_multilingual_v2"
)

# --- TTS Cache ---
tts_cache = {}


def tts_cache_key(text):
    return hashlib.sha256(text[:10].encode()).hexdigest()


# --- Conversational Fillers ---
FILLER_PHRASES = [
    "एक सेकंड जी, मैं देख रहा हूँ",
    "थोड़ी देर प्रतीक्षा करें",
    "मैं अभी चेक कर रहा हूँ",
    "एक क्षण रुकिए",
    "बस एक पल दीजिए",
    "जानकारी लोड हो रही है",
    "थोड़ा समय दीजिए",
    "मैं पुष्टि कर रहा हूँ",
    "सिस्टम से डेटा निकाल रहा हूँ",
    "मैं अभी देखता हूँ"
]

# --- Filler Audio Path ---
FILLER_AUDIO_PATH = "static/fillers"
os.makedirs(FILLER_AUDIO_PATH, exist_ok=True)

# --- Pre-generated Filler Audio ---
FILLER_AUDIO = {}


def initialize_fillers():
    """Load pre-generated filler audio from files"""
    for i, phrase in enumerate(FILLER_PHRASES):
        filename = f"filler_{i}.mp3"
        file_path = os.path.join(FILLER_AUDIO_PATH, filename)

        if not os.path.exists(file_path):
            logger.info(f"Generating filler audio for: {phrase}")
            try:
                audio = eleven_client.text_to_speech.convert(
                    text=phrase,
                    voice_id=voice_id,
                    voice_settings=VOICE_SETTINGS,
                    model_id=VOICE_SETTINGS.optimizer
                )
                with open(file_path, "wb") as f:
                    for chunk in audio:
                        if isinstance(chunk, bytes):
                            f.write(chunk)
            except Exception as e:
                logger.error(f"Error generating filler audio: {str(e)}")
                continue

        try:
            with open(file_path, "rb") as f:
                FILLER_AUDIO[phrase] = f.read()
                logger.info(f"Loaded filler audio: {phrase}")
        except Exception as e:
            logger.error(f"Error loading filler audio: {str(e)}")


initialize_fillers()

TRAINING_SCRIPT = """
Script:
Aap Airtel se Pankaj bol rahe hain aur customer ka naam Anas Ansari hai. Aapko current time ke hisaab se Good Morning / Afternoon / Evening bolna hai.

Script follow kijiye strict manner mein:
Follows exact and right flow of call as real call and use the same responses in FAQ and Script.

Agent: Good Morning / Good Afternoon / Good Evening Sir!
Agent: Main Pankaj bol raha hoon Airtel se. Kya main Anas Ansari se baat kar sakta hoon?
(Wait for customer response)

Agar customer hello bolta hai conversation ke beechmein
Agent: Hello, Kya aap mujhe sunn paa rahe hain!

Agar customer bolta hai mein recharge kardunga
Agent: Sir, aap wo kaafi time se use nahi kar rahe hain to aap device use nahi kar sakte.

Agar customer bole suspend ni hua device to chal raha hai mera
Agent: Sir, jesa mein check kar pa raha hu apne data mein apka device 3 months se zyada ho gaya use naa hote hue jiski wajah se wo suspend kardiya gaya hai or ab aap use recharge nahi kar sakte.

Agar customer ki jagah koi or baat kar raha ho or bole ki apko unse kya baat karni hai:
Agent: Sir, unhone Airtel ki services use kari thi. Ab unka device suspend ho gaya hai kyunki recharge kaafi time se nahi hua hai. Company policy ke hisaab se unhe yeh device return karna hoga. Kya aap unko batayenge ki device pickup karne ke liye Airtel se phone aaya tha taaki hum yaha se pickup arrange kar sake.

Agent ko puchna hai meri baat kisse ho rhi hai

Agar yeh Han bolte hain ki Ok mein unhe batadunga or btate hain wo kon bol rahen hain.
Agent: Dhanyevad sir, mein aasha karunga ki wi jaldi hume phone karen. Airtel mein aapka samay dene ke liye dhanyavaad Sir, aapka samay shubh ho.

Agar customer(Anas Ansari) haan kehta hai ki boliye kya baat karni hai:
Agent: Sir, aapka Airtel device recharge na hone ke kaaran suspend ho gaya hai. Or ye device company policy ke mutabiq device return karna hoga.

Agar customer device return karle ke liye agree karta hai
Agent: Kripya pickup ke liye suitable time bataye, taki main pickup arrange kar saku ?

Agar customer suitable time btadeta hai
Agent: Kya aap mujhe apna address confirm kara sakte hain? Apka current address hai 177, Udyog Vihar Phase 1, Udyog Vihar, Sector 20, Gurugram

Agar customer bolta hai nahi mein iss address par ab nahi rehta 
Agent: Shukriya, ye baat batane ke liye. Kya aap mujhe apna current address bata sakte hain.

Agar customer ready hai:
Agent: Sir, aap apna address aur pin code confirm kar dijiye please, aur pickup ki date bhi confirm kar dijiye.

Agar customer return nahi karna chahta:
Agent: Sir, yeh device Airtel ka device hai, jo aapko service ke benefit ke liye diya gaya tha. Agar aap use nahi kar rahe, toh please device humein wapस karna hoga. Yeh detail device pe bhi likha hota hai.

Agar customer return ke liye agree hota hai
Agent: Sir, kya aap mujhe apna address confirm karwa sakte hain. Taaki pickup boy asaani se apke address par aa jaega.

Agar customer puchta hai mujhe kya kya jama karna hai.
Agent: Sir, apko sirf box jama karna hai unplugged karke.

Agar customer bole ki mujhe refund milega ya ni 
Agent needs to convince them according to why refund is not possible as per given context.

Lekin device to free tha na ?
Agent: Apki jankari ke liye device free nahi hai, jo paise aapne diye the wo installation, wire, recharge ke the. Box company ka device hai, isliye uska return karna aanivarye hai. 

Consent ke liye:
Agent: Sir, main aapki details confirm kar deta hoon — aapka address hai "177, Udyog Vihar Phase 1, Udyog Vihar, Sector 20, Gurugram, Haryana 122008" aur yeh aapka registered number hai. Pickup ke din par available rahiyega, hamara executive pickup ke liye ayega.

Last message with everyone:
Agent: Kya main aapki kisi aur tarah se help kar sakta hoon?

Jab customer agree karle device return karne ke liye or sb confirm karde time, date, address then last mein jab closing message hoga to wo ye hoga
Agent: Airtel mein aapka time dene ke liye dhanyavaad Sir, aapka samay shubh ho.

Call Closing for everyone:
Agent: Airtel mein aapka time dene ke liye dhanyavaad Sir, aapka samay shubh ho.

Agar customer bole: "Kya aap AI ho?"
Agent: Nahi Sir, main real human hoon, Airtel ke calling department se baat kar raha hoon. Agar aapko doubt ho toh main aapki poori details confirm karke bata sakta hoon.

Agar recharge wapas karne ko bole customer.
Agent: Nahi sir, company ke anusar agar aap recharge kar bhi lete hain to device ni chalega wo apko return karna hi padega.

Pickup date or time confirm karte hue address bhi puche or iss address, date or time ko memory m store kare taaki customer puche to wo use bata sake.

Always remember to use convince skills not in rude manner.

Always remember if apart from customer anyother person pick the call then also we need to collect all the data from the person on the behalf of Anas Ansari.
Need to give response only from Script and FAQ's.
Always remember to introduce youreself onetime only and after that you need to convince customer to return the device as per the given context. 
ONT means wifi router and STB means setup box.
Device ke liye Box word use kre.
Don't use any context apart from this.
Customer ki unavailability jisse baat hogi usse ending m last message and clossing message bhi use krna hai.
ONT means wifi router and STB means setup box.
Device ke liye Box word use kre.
Don't use any context apart from this.
Give me the same response as in Script and FAQ if user input is related to any question and don't need to generate or rephrase from the given context. 
Act as real human no need to tell that you are an AI and if user start conversation which is not relevant then give them answer and also tell them why you call them on behalf of airtel device return.
Always remember to give the final response is correct Hindi Conversation and in correct word for pronunciation.
Based on the above context we need to start the conversation with customer no need to generate reply which is not provided in the cotext.
Give me final response in short length for time latency.
We need to follow the correct flow of device return and if user use any abusive language then we need to convince them not to use this kind of language.
Give me all response in correct hindi word and also use correct hindi pronunciation.
If there is any number then you need to speak the number in correct manner.
""".strip()


FAQ_SECTION = """
1. Q: Mujhe STB ya ONT kyu jama karna hai?
   A: Sir, device kaafi time se recharge nahi hua hai isliye suspend ho gaya hai. Company policy ke hisaab se agar device use nahi ho raha toh return karna hota hai.
2. Q: Installation aur suspension kab hua?
   A: Sir, installation ka exact date system mein nahi hota. Lekin ONT turant band ho jata hai recharge khatam hone ke baad, aur STB lagbhag 90 din baad suspend hota hai.
3. Q: Pickup boy Airtel ka ID card leke aayega kya?
   A: Sir, pickup boy Airtel ka direct employee nahi hota, wo hamare courier partner ka hota hai. Uske paas ek message aur OTP hoga, jisko verify karke aap device handover kar sakte hain.
4. Q: Kya pickup boy slip dega?
   A: Nahi Sir, wo koi physical slip nahi dega. Aapko ek message milega jisme OTP aur pickup confirmation hota hai.
5. Q: Aapka number spam dikha raha hai, kya yeh fraud ho sakta hai?
   A: Sir, hum Airtel ke behalf pe call kar rahe hain. Aapka naam, number, device ID sab hamare system mein hai. Kai users number block kar dete hain isliye spam show ho sakta hai, par aap tension mat lijiye.
6. Q: Aapke system mein mera address hona chahiye, phir bhi aap kyun puchh rahe ho?
   A: Sir, system mein address hota hai but confirmation ke liye aapka current address puchhna padta hai. Agar kuch update karna hoga toh aap bata dijiye.
7. Q: Mujhe mail bhejo, tabhi main device jama karunga.
   A: Sir, ye calling department hai, hum mail nahi bhejte. Jab pickup boy aayega tab aapko ek message milega jisme OTP aur pickup ka detail hoga.
8. Q: Mera ONT suspend ho gaya, kya ab recharge nahi hoga?
   A: Nahi Sir, device suspend ho gaya hai toh recharge nahi hoga. Company ko device return karna hoga.
9. Q: STB recharge karna hai, customer care number do.
   A: Sir, recharge aap kara sakte hain. Lekin customer care number ke liye aap Airtel ka official website ya Google se check kar lijiye.
10. Q: Mera customer ID ya landline number batao.
    A: Sir, ye details humare system mein nahi hoti. Main aapko aapka device ID ya STB/ONT number de sakta hoon.
11. Q: Jab maine product liya tha paise diye the, wo paise return honge kya?
    A: Sir, jo paise aapne diye the wo installation, wire, recharge ke the. Box company ka device hai, isliye uska return karna aanivarye hai.
12. Q: Apko kya baat krni hai unse ?
    A: Sir mein jaan sakta hun aap kon baat kar rahe hain? ( after user response bot reply:- Sir kya aaap unse meri baat kara sakte hain ? (after this response:- Sir unke paas Airtel ki taraf se ek box lia hua tha jo suspend ho gaya hai. To sir mein jaan sakta hun ki kya apke paas wo box hai ya nahi hai agar hai to kya aap wo jama karwa sakte hain.)
13. Q: Nahi mein iss address par ab nahi rehta.
    A: To sir kya aap apna new address de sakte hain taaki mein apne system m use update krdeta hu taaki delivery boy easily pickup karle.
14. Q: Nahi mein iss address par ab nahi rehta.
    A: sir kya aap apna new address de sakte hain taaki mein apne system m use update krdeta hu taaki delivery boy easily pickup karle.
15. Q: Mujhe kya kya jama karna hoga ?
    A: Sir aap box ko unplugged karke jama karna hoga. Hamara pickup boy apke paas aega apko use bas box jama karana hai, wire ya or kuch nahi karana.
16. Q: Message before call cut.
    A: Apna keemti samay dene ke liye dhannewaad. Apka samay shubh ho.
ONT means wifi router and STB means setup box.
Device ke liye Box word use kre.
Don't use any context apart from this.
Always remember to give the final response is correct Hindi Conversation and in correct word for pronunciation.
Based on the above context we need to start the conversation with customer no need to generate reply which is not provided in the cotext.
Give me final response in short length for time latency.
We need to follow the correct flow of device return and if user use any abusive language then we need to convince them not to use this kind of language.
Give me all response in correct hindi word and also use correct hindi pronunciation
""".strip()

def streaming_transcribe(audio_generator):
    config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="hi-IN",
            enable_automatic_punctuation=True,
            model="phone_call"
        ),
        interim_results=True,
        single_utterance=True
    )

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in audio_generator)

    responses = speech_client.streaming_recognize(config=config, requests=requests)

    transcript = ""
    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript += result.alternatives[0].transcript
    return transcript


def generate_voice_response(text, call_sid):
    try:
        cache_key = tts_cache_key(text)
        if cache_key in tts_cache:
            logger.info("TTS cache hit.")
            stream_id = tts_cache[cache_key]
        else:
            logger.info("TTS cache miss. Generating new audio.")
            stream_id = f"{call_sid}_{uuid.uuid4().hex}"

            audio = eleven_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                voice_settings=VOICE_SETTINGS,
                model_id=VOICE_SETTINGS.optimizer
            )
            audio_streams[stream_id] = []
            for chunk in audio:
                if isinstance(chunk, bytes):
                    audio_streams[stream_id].append(chunk)
            tts_cache[cache_key] = stream_id
        return f"{public_url}/stream/{stream_id}"
    except Exception as e:
        logger.debug(f"Voice generation error: {str(e)}")
        return None


def generate_ai_response(prompt, call_sid):
    try:
        if call_sid not in conversation_history:
            full_prompt = f"{TRAINING_SCRIPT}\n\n{prompt[:100]}\n\n{FAQ_SECTION}"
            conversation_history[call_sid] = [
                {"role": "system", "content": full_prompt},
                {"role": "assistant",
                 "content": "नमस्ते, मैं पंकज बोल रहा हूँ <break time='200ms'/> एयरटेल से। क्या मेरी बात अनस अंसारी से हो रही है?"}
            ]
        conversation_history[call_sid].append({"role": "user", "content": prompt})
        messages = [conversation_history[call_sid][0]] + conversation_history[call_sid][-4:]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=250,
            stop=["\n\n"]
        )
        ai_response = response.choices[0].message.content.strip()
        conversation_history[call_sid].append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        return f"DEBUG ERROR: {str(e)}"


def get_random_filler():
    filler = random.choice(FILLER_PHRASES)
    audio_data = FILLER_AUDIO.get(filler, b"")
    return filler, audio_data


@app.route('/stream/<path:stream_id>')
def audio_stream(stream_id):
    file_path = os.path.join("audio_cache", stream_id)
    if os.path.exists(file_path):
        return send_file(file_path)

    def generate():
        if stream_id in audio_streams:
            for chunk in audio_streams[stream_id]:
                yield chunk
            del audio_streams[stream_id]

    return Response(generate(), mimetype='audio/mpeg')


@app.route('/filler/<call_sid>')
def filler_audio(call_sid):
    """Returns a single filler audio phrase"""
    if call_sid.startswith("CA"):
        time.sleep(0.3)

    filler_text, audio_data = get_random_filler()
    logger.info(f"Playing filler: {filler_text}")
    return Response(audio_data, mimetype='audio/mpeg')


processing_lock = threading.Lock()


@app.route("/voice", methods=["GET", "POST"])
def voice():
    resp = VoiceResponse()
    call_sid = request.values.get("CallSid")
    if not call_sid:
        return str(resp)

    try:
        start_time = time.time()
        if call_sid not in conversation_history:
            full_prompt = f"{TRAINING_SCRIPT}\n\nUser: Hi\n{FAQ_SECTION}"
            conversation_history[call_sid] = [
                {"role": "system", "content": full_prompt},
                {"role": "assistant",
                 "content": "नमस्ते, मैं पंकज बोल रहा हूँ <break time='200ms'/> एयरटेल से। क्या मेरी बात अनस अंसारी से हो रही है?"}
            ]
            ai_response = conversation_history[call_sid][-1]["content"]

            tts_start = time.time()
            stream_url = generate_voice_response(ai_response, call_sid)
            tts_end = time.time()
            logger.info(f"TTS latency: {tts_end - tts_start:.2f}s")

            with resp.gather(
                    input='speech',
                    speech_timeout='auto',
                    barge_in=True,
                    action='/voice',
                    method='POST',
                    timeout=300,
                    language='hi-IN'
            ) as gather:
                if stream_url:
                    gather.play(stream_url)
                else:
                    gather.say(ai_response, language='hi-IN')

        elif 'SpeechResult' in request.form:
            user_input = request.form['SpeechResult']
            logger.info(f"User said: {user_input}")

            processing_status[call_sid] = {
                "status": "processing",
                "filler_played": False
            }

            def process_response():
                try:
                    ai_response = generate_ai_response(user_input, call_sid)
                    logger.info(f"AI response: {ai_response}")

                    tts_start = time.time()
                    stream_url = generate_voice_response(ai_response, call_sid)
                    tts_end = time.time()
                    logger.info(f"TTS latency: {tts_end - tts_start:.2f}s")

                    with processing_lock:
                        with processing_lock:
                            processing_status[call_sid] = {
                                "status": "complete",
                                "response": ai_response,
                                "stream_url": stream_url
                            }
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    processing_status[call_sid] = {
                        "status": "error",
                        "error": str(e)
                    }

            threading.Thread(target=process_response).start()

            filler_url = f"{public_url}/filler/{call_sid}"
            resp.play(filler_url)
            processing_status[call_sid]["filler_played"] = True

            resp.redirect(f"{public_url}/voice?call_sid={call_sid}", method='POST')

        else:
            status = processing_status.get(call_sid, {})

            if isinstance(status, dict) and status.get("status") == "processing":
                resp.pause(3)
                resp.redirect(f"{public_url}/voice?call_sid={call_sid}", method='POST')

            elif isinstance(status, dict) and status.get("status") == "complete":
                ai_response = status["response"]
                stream_url = status["stream_url"]

                with resp.gather(
                        input='speech',
                        speech_timeout='auto',
                        barge_in=True,
                        action='/voice',
                        method='POST',
                        timeout=5,
                        language='hi-IN'
                ) as gather:
                    if stream_url:
                        gather.play(stream_url)
                    else:
                        gather.say(ai_response, language='hi-IN')

                if call_sid in processing_status:
                    del processing_status[call_sid]

            elif isinstance(status, dict) and status.get("status") == "error":
                error_msg = "तकनीकी समस्या हो रही है। कृपया दोबारा बोलें।"
                stream_url = generate_voice_response(error_msg, call_sid)
                with resp.gather(
                        input='speech',
                        speech_timeout='auto',
                        barge_in=True,
                        action='/voice',
                        method='POST',
                        timeout=5,
                        language='hi-IN'
                ) as gather:
                    if stream_url:
                        gather.play(stream_url)
                    else:
                        gather.say(error_msg, language='hi-IN')

                if call_sid in processing_status:
                    del processing_status[call_sid]

            else:
                ai_response = "सर, क्या आप बता सकते हैं कि डिवाइस पिकअप के लिए सही समय क्या होगा?"
                stream_url = generate_voice_response(ai_response, call_sid)

                with resp.gather(
                        input='speech',
                        speech_timeout='auto',
                        barge_in=True,
                        action='/voice',
                        method='POST',
                        timeout=5,
                        language='hi-IN'
                ) as gather:
                    if stream_url:
                        gather.play(stream_url)
                    else:
                        gather.say(ai_response, language='hi-IN')

        end_time = time.time()
        logger.info(f"Total /voice handler latency: {end_time - start_time:.2f}s")

    except Exception as e:
        logger.error(f"Voice handler error: {str(e)}")
        resp.say("तकनीकी समस्या है। कृपया बाद में कॉल करें।", language='hi-IN')

    return str(resp)


if __name__ == "__main__":
    from waitress import serve

    server_thread = threading.Thread(
        target=serve,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 5000},
        daemon=True
    )
    server_thread.start()

    time.sleep(0.5)

    try:
        logger.info(f"Calling {recipient_number}")
        call = twilio_client.calls.create(
            url=f"{public_url}/voice",
            to=recipient_number,
            from_=twilio_phone_number,
            machine_detection='Enable'
        )
        logger.info(f"Call started: {call.sid}")
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Call failed: {str(e)}")