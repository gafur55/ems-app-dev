#!bin bash

cd /Users/gafurmammadov/Documents/Uchicago_classes/practicum/ems_lib/ems-main/ems/app_dev

if command -v deactivate &> /dev/null; then
    deactivate
fi

rm -rf .venv

python3 -m venv .venv

source .venv/bin/activate

pip3 install -r requirements.txt

python3 main.py