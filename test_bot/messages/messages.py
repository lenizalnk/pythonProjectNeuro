from aiogram.types import ReplyKeyboardMarkup

msgs = {
    "hello": "Здравствуйте! Какой у вас вопрос?",
    "intent_do": "БАЗА знаний делаете",
    "intent_can": "можете",
    "intent_chair": "обивка",
    "intent_change": "поменять",
    "intent_repair": "сделаете",
    "intent_none": "Я вас не понял"
}

main_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add("Вывести список пицц")
main_keyboard.insert("Повторить предыдущий заказ")