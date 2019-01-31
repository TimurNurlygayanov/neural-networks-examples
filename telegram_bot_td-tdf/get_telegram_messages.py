#!/usr/bin/python3
# -*- encoding=utf8 -*-

from telethon.sync import TelegramClient
from configparser import ConfigParser

import sys
sys.path.append('../')
from spell_checker_library.spell_checker import Checker


def get_conf_param(parameter, default_value):
    """ This function reads and returns the value of parameter
        from configuration file.
    """

    result = config.get('DEFAULT', parameter)
    return result or default_value


my_checker = Checker()

# Read Telegram private key from the local file:
config = ConfigParser()
config.read('/Users/timurnurlygayanov/.config.ini')


# Read all parameters from config file:
name = get_conf_param('name', '')
api_id = get_conf_param('api_id', '')
api_hash = get_conf_param('api_hash', '')
chat = get_conf_param('chat', '')

ALL_MESSAGES = []
ALL_QUESTIONS = []

with TelegramClient(name, api_id, api_hash) as client:
    for message in client.iter_messages(chat, limit=4000):
        if message.text:
            ALL_MESSAGES.append(str(message.text))

for q in ALL_MESSAGES:
    msg = q.replace('.', '\n').replace('(', '\n').replace(')', '\n')
    msgs = msg.replace('"', '\n').lower().split('\n')

    for m in msgs:
        if len(m) > 5:
            if '?' in str(m):
                for question in m.split('?')[:-1]:
                    ALL_QUESTIONS.append(question + '?')


"""
my_checker.learn('\n'.join(ALL_MESSAGES))


for i, q in enumerate(ALL_QUESTIONS):
    q = str(q).strip()
    q2 = my_checker.spellcheck(q)

    ALL_QUESTIONS[i] = q2
"""

ALL_QUESTIONS = [q.strip() for q in ALL_QUESTIONS if ' ' in q and len(q) > 30]

print('Total messages: {0}'.format(len(ALL_MESSAGES)))
print('Questions found: {0}'.format(len(ALL_QUESTIONS)))

with open('messages.txt', 'w') as f:
    f.writelines('\n'.join(ALL_MESSAGES))

with open('questions.txt', 'w') as f:
    f.writelines('\n'.join(ALL_QUESTIONS))
