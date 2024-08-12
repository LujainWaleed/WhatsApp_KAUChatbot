import json
import numpy as np
import pickle
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class Chatbot:
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def __init__(self, intents, model, tokenizer, max_length):
        self.intents = intents
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def start_chat(self):
        print("Chatbot is ready to talk! Type 'quit' to exit.")
        while True:
            try:
                user_response = input("You: ").strip()
                if self.make_exit(user_response):
                    print("Have a nice day!")
                    break
                response = self.generate_response(user_response)
                print(f"Bot: {response}")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nChat interrupted. Exiting...")
                break

    def make_exit(self, user_response):
        return user_response.lower() in self.exit_commands

    def generate_response(self, user_response):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                if pattern.lower() in user_response.lower():
                    response = random.choice(intent['responses'])
                    return response

        stemmed_sentence = self.get_stemmed_sentence(user_response)
        tag = self.get_predicted_tag_class(stemmed_sentence)
        reverse_dict = self.get_reverse_dict()
        tag_name = reverse_dict.get(tag, "unknown")

        logging.info(f"Predicted Tag: {tag}")
        logging.info(f"Tag Name: {tag_name}")

        for intent in self.intents['intents']:
            if intent['tag'] == tag_name:
                response = random.choice(intent['responses'])
                return response

        return "Sorry, I didn't understand that."

    def get_predicted_tag_class(self, stemmed_sentence):
        seq = self.tokenizer.texts_to_sequences([stemmed_sentence])
        pad_seq = pad_sequences(seq, maxlen=self.max_length, padding='post')
        x = self.model.predict(pad_seq)
        y = np.argmax(x)
        return y

    def get_reverse_dict(self):
        tags = [intent['tag'] for intent in self.intents['intents']]
        reverse_dict = {i: tag for i, tag in enumerate(tags)}
        return reverse_dict

    def get_stemmed_sentence(self, sentence):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(sentence)
        filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
        return ' '.join(filtered_tokens)


app = Flask(__name__)

# Initialize state management
user_state = {}
user_previous_state = {}

def show_list(state):
    """Return appropriate message for the given state"""
    if state == "initial": #it will change
        return "اختر نوع الابتعاث:\n1- داخلي\n2- خارجي\n3- الرجوع للقائمة السابقة"
    elif state == "master's_level":
        return "اختر نوع الابتعاث:\n1- داخلي\n2- خارجي\n3- الرجوع للقائمة السابقة"
    elif state == "internal_master's_scholarship":
        return "اختر نوع الاستفسار:\n1- اسئلة قبل الابتعاث\n2- اجراءات اثناء الابتعاث\n3- اجراءات انهاء الابتعاث\n4- الرجوع"
    elif state == "Pre questions":
        return "\nأول خطوات ما قبل الابتعاث هي الحصول على ضمان مالي.\nاختر مما يلي:\n1- تم الحصول على الضمان المالي\n2- لم يتم الحصول على ضمان مالي بعد \n3- رجوع" #رجوع
    elif state == "external_master's_scholarship":
        return "اختر نوع الاستفسار:\n1- اسئلة قبل الابتعاث\n2- اجراءات اثناء الابتعاث\n3- اجراءات انهاء الابتعاث\n4- الرجوع"
    elif state == "phd_level":
        return "اختر نوع الابتعاث:\n1- داخلي\n2- خارجي\n3- الرجوع للقائمة السابقة"
    elif state == "internal_phd_scholarship":
        return "اختر نوع الاستفسار:\n1- اسئلة قبل الابتعاث\n2- اجراءات اثناء الابتعاث\n3- اجراءات انهاء الابتعاث\n4- الرجوع"
    elif state == "external_phd_scholarship":
        return "اختر نوع الاستفسار:\n1- اسئلة قبل الابتعاث\n2- اجراءات اثناء الابتعاث\n3- اجراءات انهاء الابتعاث\n4- الرجوع"
    elif state == "not yet":
        return"\nفي حال عدم الحصول على ضمان مالي لديك نوعان من الضمان المالي: \n1- ضمان مالي للبحث عن قبول\n2- ضمان مالي للتقديم على تأشيرة\n3- رجوع " #رجوع
    elif state == "during questions":
        return "قائمة اثناء البعثة :\n1- طلب تغيير تخصص/جامعة\n2- طلب حضور مؤتمرات/رحلات\n3-طلب ترقية البعثة\n4- طلب تمديد البعثة\n5- رجوع الى القائمة السابقة"
    elif state == "change major/uni":
        return "قائمة تغيير تخصص /جامعه:\n1. ماقبل تقديم طلب تغيير تخصص\n2. اجراءات تقديم طلب تخصص/ جامعه\n3. رجوع الى القائمة السابقة"
    elif state == "trips":
        return "قائمة طلب حضر مؤتمرات /رحلات:\n1. ماقبل تقديم طلب حضور مؤتمرات/رحلات\n2. اجراءات تقديم طلب حضور مؤتمرات ,ندوات,دورات قصيرة\n3. اجراءات تقديم طلب حضور رحلات\n4. رجوع الى القائمة السابقة"
    elif state == "sci/short trips":
        return "قائمة ماقبل تقديم طلب حضور مؤتمرات/رحلات:\n1. ماقبل طلب رحله علمية\n2. ماقبل طلب حضور مؤتمرات, ندوات, دورات قصيرة\n3. رجوع"
    elif state == "Upgrade of mission":
        return "\nقائمة ترقية البعثة:\n1- انواع الترقية\n2- اجراءات تقديم طلب الترقية\n3- الرجوع إلى القائمة السابقة"
    elif state == "Extention year":
        return "قائمة التمديد:\n1- اجراءات التمديد\n2- رجوع\n"
    elif state == "After questions":
        return "\nما بعد الابتعاث:\n1- الحصول على الدرجة\n2- عدم الحصول على درجة\n3- الرجوع"


    else:
        return "حالة غير معروفة."

def incoming_sms(msg, from_number):
    """Handle messages routed to SMS"""
    logging.info(f"Processing SMS message: {msg} from {from_number}")

    resp = MessagingResponse()

    # Determine user state based on phone number
    state = user_state.get(from_number, "initial")
    prev_state = user_previous_state.get(from_number, None)

    if msg:
        msg = msg.strip().lower()

        if state == "initial":
            if msg == '1':
                user_previous_state[from_number] = state
                user_state[from_number] = "master's_level"
                resp.message(show_list("master's_level"))
            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "phd_level"
                resp.message(show_list("phd_level"))
            elif msg == '3':
                resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "master's_level":
            if msg == '1':
                user_previous_state[from_number] = state
                user_state[from_number] = "internal_master's_scholarship"
                resp.message(show_list("internal_master's_scholarship"))
            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "external_master's_scholarship"
                resp.message(show_list("external_master's_scholarship"))
            elif msg == '3':
                user_state[from_number] = "initial"
                resp.message(
                    "*أهلاً بك في مساعد الابتعاث بجامعة الملك عبدالعزيز!*\n\nيسعدني أن أكون رفيقك في رحلة الابتعاث. أنا هنا لتقديم الدعم والمعلومات التي تحتاجها في كل خطوة من خطوات رحلتك.\n\nيرجى اختيار الدرجة العلمية التي تود الاستفسار عنها:\n1- ماجستير\n2- دكتوراه\n\nأو يمكنك البدء في طرح أي استفسار\n")

            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    resp.message(show_list(prev_state))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")

        elif state == "internal_master's_scholarship":
            if msg == '1': 
                user_previous_state[from_number] = state
                user_state[from_number] = "Pre questions"
                resp.message(show_list("Pre questions"))

            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "during questions"
                resp.message(show_list("during questions"))
            elif msg == '3':
                user_previous_state[from_number] = state
                user_state[from_number] = "After questions"
                resp.message(show_list("After questions"))
            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "master's_level"
                    resp.message(show_list("master's_level"))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "Pre questions":
            if msg == '1': 
                resp.message(
                    "\nفي حال الحصول على ضمان مالي باستطاعتك التقديم على طلب الابتعاث الالكتروني باتباع الخطوات التالية:\n"
                    "1- تأشيرة الدخول إلى بلد الابتعاث\n"
                    "2- الهوية الوطنية\n"
                    "3- شهادة اتمام دورة الاعداد للابتعاث\n"
                    "4- موافقة مجلس القسم العلمي والكلية\n"
                    "5- خطاب القبول من جامعة موصى بها\n"
                    "6- المؤهلات العلمية\n"
                    "7- متابعة طلب الابتعاث حتى تتك موافقة رئيس الجامعة\n"
                    "8- اصدار قرار الابتعاث التنفيذي\n"
                    "9- الحصول على تذاكر السفر عبر منصة اعتماد")
                resp.message(show_list("Pre questions"))

            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "not yet"
                resp.message(show_list("not yet"))
            elif msg == '3':

                user_state[from_number] = prev_state
                user_state[from_number] = "internal_master's_scholarship"
                resp.message(show_list("internal_master's_scholarship"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "not yet":
            if msg == '1': 

                resp.message("\nما يلي قائمة بمتطلبات الحصول على ضمان مالي للبحث عن قبول:\n"
                             "- الهوية الوطنية\n"
                             "- صورة من شهادة البكالوريوس أو الماجستير\n"
                             "- صورة درجة آخر اختبار لغة\n"
                             "- صورة من قرار التعيين بالجامعة")
                resp.message(show_list("not yet"))


            elif msg == '2':
                resp.message("\nما يلي قائمة بمتطلبات الحصول على ضمان مالي للبحث عن قبول:\n"
                             "- الهوية الوطنية\n"
                             "- صورة من شهادة البكالوريوس أو الماجستير\n"
                             "- صورة درجة آخر اختبار لغة\n"
                             "- صورة من قرار التعيين بالجامعة")
                resp.message(show_list("not yet"))
            elif msg == '3':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "Pre questions"
                    resp.message(show_list("Pre questions"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")

        elif state == "during questions":
            if msg == '1': 
                user_previous_state[from_number] = state
                user_state[from_number] = "change major/uni"
                resp.message(show_list("change major/uni"))
            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "trips"
                resp.message(show_list("trips"))
            elif msg == '3':
                user_previous_state[from_number] = state
                user_state[from_number] = "Upgrade of mission"
                resp.message(show_list("Upgrade of mission"))
            elif msg == '4':
                user_previous_state[from_number] = state
                user_state[from_number] = "Extention year"
                resp.message(show_list("Extention year"))
            elif msg == '5':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "internal_master's_scholarship"
                    resp.message(show_list("internal_master's_scholarship"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "change major/uni":
            if msg == '1': 
                resp.message(
                    "ماقبل تقديم طلب تغيير تخصص:\n- يتم تقديم الطلب قبل بدء الدراسة بثلاث الى اربعة اشهر لاتخاذ القرار.\n- ويجب على المبتعث ألا يغير تخصصه حتى يصدر قرار الموافقة بالتغيير من الجامعة، لكي لا يتعرض لإيقاف مخصصاته المالية.\n- يُوصى بالتريث في اختيار التخصص، والتشاور مع الأقسام العلمية.")
                resp.message(show_list("change major/uni"))
            elif msg == '2':
                resp.message(
                    "اجراءات تقديم طلب تغيير تخصص/جامعه:\nمنصة انجز: أولا\nيتم الرفع في منصة انجز بنوع الطلب تغيير تخصص /جامعه واجراء مايلي\n\nتغيير تخصص:\nارفاق مايلي:\n- كتابة الاسباب في منصة انجز\n- تقرير من المشرف الاكاديمي بالجامعه\n- خطاب قبول التخصص الجديد\n- ارفاق صورة من شهادة او الخطاب انها اكملت المرحلة السابقة\n\nتغيير جامعة:\nارفاق مايلي:\n- كتابة الاسباب في منصة انجز\n- تقرير من المشرف الاكاديمي بالجامعه\n- خطاب قبول التخصص الجديد\n- ارفاق صورة من شهادة او الخطاب انها اكملت المرحلة السابقة\n- تقرير من المشرف الاكاديمي بالجامعه")
                resp.message(show_list("change major/uni"))

            elif msg == '3':
                if prev_state:
                    user_state[from_number] = prev_state
                    resp.message(show_list("during questions"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "trips":
            if msg == '1': 
                user_previous_state[from_number] = state
                user_state[from_number] = "sci/short trips"
                resp.message(show_list("sci/short trips"))

            elif msg == '2':
                resp.message(
                    "اجراءات طلب حضور مؤتمرات، ندوات، دورات قصيرة\n\nأولاً منصة انجز: رفع الطلب في منصة انجز وإرفاق ما يلي: 1- صورة من كشف الدرجات الحديثة 2- خطاب من المشرف الدراسي يدعم دراسة المادة 3- صورة من الخطة الدراسية معتمدة وتبين بأن المادة متطلب الخطة الدراسية")
                resp.message(show_list("trips"))

            elif msg == '3':
                resp.message(
                    "إجراءات طلب رحلة علمية:\n\nأولاً: منصة سفير: يجب رفع طلب رحلة علمية في منصة سفير مع مراعاة المطلوب منه. وتأكد من خانة سير العمل لمعرفة المتطلبات أو النقصان في المعلومات.\n\nثانياً: منصة أنجز: بعد رفع الطلب في منصة سفير ومن ثم يتم الموافقة عليه، يتم الرفع في منصة أنجز مع وضع رقم الطلب في منصة سفير (رقم يظهر مع رفع الطلب في منصة سفير).\n\nوارفاق ما يلي: 1- توصية من المشرف الأكاديمي الرئيسي تفيد بموافقة على القيام برحلة علمية واعتماده لخطة العمل الميدانية. 2- خطاب موافقة الجهة /الجهات التي ينوي إجراء الرحلة العلمية لها إذا كانت خارج جامعة الملك عبد العزيز. 3- نسخة من الخطة المتعلقة بموضوع الرحلة العلمية موضحاً فيها الهدف الرئيسي من إجرائها والخطوات التي ستتبع لجمع مادة البحث.")
                resp.message(show_list("trips"))

            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "during questions"
                    resp.message(show_list("during questions"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "sci/short trips":
            if msg == '1':  
                resp.message(
                    "الرحلة العلمية هي عبارة عن عمل ميداني يقوم به المبتعث مرة واحدة خلال المرحلة الدراسية الواحدة لجمع بعض البيانات المتعلقة ببحثه وأطروحته. يجوز للمبتعث إلى الخارج القيام برحلة علمية أثناء إعداد الرسالة ولمرة واحدة خلال المرحلة الدراسية الواحدة إلى المملكة أو غيرها خارج مقر البعثة وفقاً للضوابط الآتية: 1 - أن يوصي المشرف على دراسة الطالب بحاجة البحث إلى الرحلة العلمية. 2 - موافقة مجلس القسم والكلية، أو المعهد وما في حكمهما، ولجنة الابتعاث والتدريب في الجامعة المبتعث منها. 3 - ألا تزيد مدة الرحلة العلمية عن ثلاثة أشهر حداً أقصى.")
                resp.message(show_list("sci/short trips"))
            elif msg == '2':
                resp.message(
                    "ماقبل طلب حضور مؤتمرات ,ندوات,دورات قصيرة يصرف للمبتعث تذكرة سفر ذهاباً وإياباً لمرة واحدة لحضور المؤتمرات، والندوات العلمية، أو الدورات القصيرة وذلك خلال المرحلة الدراسية الواحدة وفق الضوابط الآتية: - أن يكون للمؤتمر، أو الدورة علاقة مباشرة بتخصصه أو موضوع بحثه. - موافقة لجنة الابتعاث والتدريب في الجامعة بناء على توصية المشرف على دراسة الطالب. ففي حالة رغبة المبتعث المشاركة في مؤتمر علمي أو دورة تدريبية قصيرة فعليه تقديم طلب بذلك يوضح فيه عنوان المؤتمر أو الدورة وعلاقته بتخصصه أو موضوع بحثه ومكان انعقاده سواء كان ذلك في بلد الابتعاث أو بلد آخر ، وموعد انعقاده والفترة التي يستغرقها . وإرفاق النشرات التعريفية بذلك ، على أن يكون التقديم قبل موعد إقامة المؤتمر أو الدورة بوقت كافٍ لا يقل عن شهرين لتتمكن الجهات المختصة بالملحقية والجامعة من البت في الموضوع.")
                resp.message(show_list("sci/short trips"))
            elif msg == '3':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "trips"
                    resp.message(show_list("trips"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "Upgrade of mission":
            if msg == '1': 
                resp.message(
                    "\n انواع الترقية:\nماجستير -> دكتوراه\nدكتور في المجال الطبي -> زمالة\n*- يتم رفع طلب ترقية بعد الانتهاء من المرحلة الدراسية وبدء مرحلة دراسية جديدة*")
                resp.message(show_list("Upgrade of mission"))

            elif msg == '2':
                resp.message(
                    "اجراءات تقديم طلب الترقية:\nيتم الانتهاء من الفترة الدراسية والحصول على الشهادة العلمية وعمل الآتي:\n\nأولاً: منصة سفير:\n رفع طلب الترقية في منصة سفير أولاً مع مراعاة المطلوب منه. وتأكد من خانة سير العمل لمعرفة المتطلبات أو النقصان في المعلومات.\n\nثانياً: منصة أنجز:\nبعد الموافقة على طلب الترقية في منصة سفير، يتم رفعه في منصة أنجز مع وضع رقم الطلب في منصة سفير (رقم يظهر مع رفع الطلب في منصة سفير).\n\nوارفاق مايلي:\n1- تقرير بما تم إنجازه خلال الفترة السابقة.\n2- خطاب قبول المرحلة الجديدة.\n3- خطاب يفيد باكمال المرحلة السابقة بنجاح.")
                resp.message(show_list("Upgrade of mission"))

            elif msg == '3':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "during questions"
                    resp.message(show_list("during questions"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "Extention year":
            if msg == '1': 
                resp.message("*حسب لائحة الابتعاث و التدريب فإنه سيسمح لمبتعثي الدكتوراه بثلاث تمديدات, و الماجستير بتمديدين.\n*حسب قرار مجلس شؤون الجامعات الجديد بتاريخ 17/12/1445, مدة التمديد للدكتوراه و الماجستير واحدة فقط.")
                resp.message(show_list("Extention year"))
            elif msg == '2':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "during questions"
                    resp.message(show_list("during questions"))

            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")

        elif state == "After questions":
            if msg == '1':
                resp.message("\nفي حال الحصول على الدرجة العلمية:\n"
                             "- تقديم طلب إنهاء دراسة على نظام سفير\n"
                             "- تقديم طلب تخرج على نظام أنجز\n"
                             "- خطاب من الجامعة بالحصول على الدرجة العلمية\n"
                             "- إرفاق نسخة من كشف الدرجات والشهادات التي يحصل عليها المبتعث\n"
                             "- تقديم طلب مباشرة على نظام أنجز ومباشرة العمل فعليًا لحين صدور قرار التخرج التنفيذي")
                resp.message(show_list("After questions"))
            elif msg == '2':
                resp.message("\nالإجراءات اللازمة في حال عدم الحصول على درجة:\n"
                             "- تقديم طلب انهاء بعثة على نظام أنجز\n"
                             "- المستندات الخاصة بمبررات الإنهاء\n"
                             "- رفع بدل الطباعة\n"
                             "- تقديم طلب المباشرة على نظام أنجز و مباشرة العمل فعليًا لحين صدور قرار الإنهاء التنفيذي")
                resp.message(show_list("After questions"))
            elif msg == '3':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "internal_master's_scholarship"
                    resp.message(show_list("internal_master's_scholarship"))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "external_master's_scholarship":
            if msg == '1':  
                user_previous_state[from_number] = state
                user_state[from_number] = "Pre questions"
                resp.message(show_list("Pre questions"))

            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "during questions"
                resp.message(show_list("during questions"))
            elif msg == '3':
                user_previous_state[from_number] = state
                user_state[from_number] = "After questions"
                resp.message(show_list("After questions"))
            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    resp.message(show_list(prev_state))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "phd_level":
            if msg == '1':
                user_previous_state[from_number] = state
                user_state[from_number] = "internal_phd_scholarship"
                resp.message(show_list("internal_phd_scholarship"))
            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "external_phd_scholarship"
                resp.message(show_list("external_phd_scholarship"))
            elif msg == '3':
                user_state[from_number] = "initial"
                resp.message("*أهلاً بك في مساعد الابتعاث بجامعة الملك عبدالعزيز!*\n\nيسعدني أن أكون رفيقك في رحلة الابتعاث. أنا هنا لتقديم الدعم والمعلومات التي تحتاجها في كل خطوة من خطوات رحلتك.\n\nيرجى اختيار الدرجة العلمية التي تود الاستفسار عنها:\n1- ماجستير\n2- دكتوراه\n\nأو يمكنك البدء في طرح أي استفسار\n")

            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    resp.message(show_list(prev_state))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "internal_phd_scholarship":
            if msg == '1':
                user_previous_state[from_number] = state
                user_state[from_number] = "Pre questions"
                resp.message(show_list("Pre questions"))

            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "during questions"
                resp.message(show_list("during questions"))
            elif msg == '3':
                user_previous_state[from_number] = state
                user_state[from_number] = "After questions"
                resp.message(show_list("After questions"))
            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "master's_level"
                    resp.message(show_list("master's_level"))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")
        elif state == "external_phd_scholarship":
            if msg == '1':  
                user_previous_state[from_number] = state
                user_state[from_number] = "Pre questions"
                resp.message(show_list("Pre questions"))

            elif msg == '2':
                user_previous_state[from_number] = state
                user_state[from_number] = "during questions"
                resp.message(show_list("during questions"))
            elif msg == '3':
                user_previous_state[from_number] = state
                user_state[from_number] = "After questions"
                resp.message(show_list("After questions"))
            elif msg == '4':
                if prev_state:
                    user_state[from_number] = prev_state
                    user_state[from_number] = "master's_level"
                    resp.message(show_list("master's_level"))
                else:
                    user_state[from_number] = "initial"
                    resp.message("أنت بالفعل في القائمة الرئيسية.")
            else:
                resp.message("خيار غير صحيح. يرجى إعادة المحاولة.")

    else:
        logging.error("No message body found in request.")
        resp.message("Sorry, I didn't understand that.")

    return str(resp)

def reply_bot(msg):
    """Handle messages routed to the bot"""
    # Initialize your chatbot (assuming these are defined elsewhere)
    chat = Chatbot(intents, model, tokenizer, max_length)
    bot_response = chat.generate_response(msg)  # Pass the actual message to the bot

    response = MessagingResponse()
    response.message(bot_response)
    return str(response)

@app.route("/")
def hello():
    return "مرحبا"

@app.route("/handle_message", methods=["POST"])
def handle_message():
    """Route incoming messages based on whether the message is a number or not"""
    msg = request.values.get('Body', None)
    from_number = request.values.get('From', None)
    logging.info(f"Received message: {msg} from {from_number}")

    if msg:
        msg = msg.strip().lower()
        if msg.isdigit():
            return incoming_sms(msg, from_number)
        else:
            return reply_bot(msg)
    else:
        logging.error("No message body found in request.")
        resp = MessagingResponse()
        resp.message("Sorry, I didn't understand that.")
        return str(resp)

if __name__ == "__main__":
    try:
        # Load intents from JSON file
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)

        # Load machine learning model
        model = load_model("ret_chatbot.h5")

        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as infile:
            tokenizer = pickle.load(infile)

        # Load max sequence length
        with open('max_seq_length', 'rb') as infile:
            max_length = pickle.load(infile)

        # Initialize and start the chatbot
        # chat = Chatbot(intents, model, tokenizer, max_length)
        # chat.start_chat()
        # Uncomment the next line if you want to test locally
        app.run(debug=True)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
