#!/bin/bash

# Verifica que se haya pasado el directorio como argumento
if [ $# -ne 1 ]; then
  echo "Uso: $0 <directorio_con_wavs>"
  exit 1
fi

input_dir=$1
output_dir="${input_dir}_44100"

# Crea el directorio de salida si no existe
mkdir -p "$output_dir"

# Recorre todos los archivos .wav en el directorio y subdirectorios
find "$input_dir" -type f -iname "*.wav" | while read -r file; do
  # Genera la ruta de salida manteniendo la estructura de carpetas
  relative_path="${file#$input_dir/}"
  output_path="$output_dir/$relative_path"
  output_dirname=$(dirname "$output_path")

  mkdir -p "$output_dirname"

  # Realiza la conversión con sox
  sox "$file" -r 44100 "$output_path"
  echo "Convertido: $file -> $output_path"
done
