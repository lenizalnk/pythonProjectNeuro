from messages import get_message_text

import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State

from aiogram.contrib.fsm_storage.files import JSONStorage

from settings import API_TOKEN

logging.basicConfig(level=logging.DEBUG)

bot = Bot(token=API_TOKEN)

storage = JSONStorage("states.json")

dp = Dispatcher(bot, storage=storage)

get_intent_callback = lambda text: "intent_not_found"


# analog
# def get_intent_callback(text):
#     return "intent_not_found"

class StateMachine(StatesGroup):
    main_state = State()


@dp.message_handler(commands=['start', 'help'], state="*")
async def send_welcome(message: types.Message):
    await StateMachine.main_state.set()
    await message.reply(get_message_text("hello"))

    logging.info(f"{message.from_user.username}: {message.text}")


@dp.message_handler(state=StateMachine.main_state)
async def main_state_handler(message: types.Message, state: FSMContext):
    intent = get_intent_callback(message.text)

    messages_from_intent = {
        "делаете": "intent_do",
        "можете": "intent_can",
        "обивка": "intent_chair",
        "поменять": "intent_change",
        "сделаете": "intent_repair",
    }

    if intent in messages_from_intent:
        await message.answer(get_message_text(messages_from_intent[intent]))
    else:
        await message.answer(get_message_text("intent_none"))

    logging.info(f"{message.from_user.username}: ({intent})  {message.text}")


def run_bot(_get_intent_callback):
    if _get_intent_callback is not None:
        global get_intent_callback
        get_intent_callback = _get_intent_callback
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    run_bot(get_intent_callback)