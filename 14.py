import logging

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

TOKEN = ""
#made by ivan

dp = Dispatcher(storage=MemoryStorage())
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

@dp.message()
async def new_msg(message: types.Message):
    shopping_list = message.text.strip().replace(' ', '').split(',')
    builder = InlineKeyboardBuilder()
    for item in range (len(shopping_list)):
        builder.add(types.InlineKeyboardButton(text=shopping_list[item], callback_data="prefix_for_this_markup:text" + str(item)))
    builder.adjust(1)
    await bot.send_message(chat_id=message.chat.id, text="Нужно купить", reply_markup=builder.as_markup())
    

@dp.callback_query()
async def get_callback(call: types.CallbackQuery):

    await call.answer()
    builder = InlineKeyboardBuilder()
    current_markup = call.message.reply_markup.inline_keyboard
    
    counter = 0

    for row in current_markup:
        for but in row:
            if but.callback_data != call.data:
                counter += 1
                builder.add (types.InlineKeyboardButton(text=but.text, callback_data=but.callback_data))
    
    builder.adjust(1)
    if counter == 0:
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        msg = await bot.send_message (chat_id=call.message.chat.id, text="Вы всё купили")
        await asyncio.sleep(10)
        await bot.delete_message(chat_id=call.message.chat.id, message_id=msg.message_id)
    else:
        await bot.edit_message_reply_markup(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                            reply_markup=builder.as_markup())


async def main():
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


logging.basicConfig(level=logging.INFO)
asyncio.run(main())