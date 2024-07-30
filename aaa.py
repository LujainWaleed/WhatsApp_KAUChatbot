import json

# تحميل البيانات من ملف JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# دالة للبحث عن الاستجابة المناسبة بناءً على المدخلات
def get_response(user_input):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            if pattern in user_input:
                return intent['responses'][0]
    return "عذرًا، لم أتمكن من فهم الطلب."

# مثال على كيفية استخدام الدالة
if __name__ == "__main__":
    while True:
        user_input = input("أدخل استفسارك: ")
        if user_input.lower() == "خروج":
            break
        response = get_response(user_input)
        print(response)