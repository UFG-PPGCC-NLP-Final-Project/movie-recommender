#!/bin/bash
# Script para executar o menu principal sem problemas do debugger

cd "$(dirname "$0")"
source venv/bin/activate
python src/main.py

