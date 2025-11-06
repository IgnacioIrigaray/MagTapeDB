#!/bin/bash

# Verifica que se haya pasado el directorio
if [ $# -ne 1 ]; then
    echo "Uso: $0 <directorio_con_wavs>"
    exit 1
fi

DIR="$1"

# Recorre todos los archivos .wav del directorio (no incluye subcarpetas)
for file in "$DIR"/*.wav; do
    [ -e "$file" ] || continue  # salta si no hay archivos .wav
    echo "Procesando: $file"
    python inference.py inference.audio="$file"
done
