import librosa
import numpy as np
import argparse

def detectar_frecuencia_y_factor(audio_path):
    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=None)

    # Estimar desviación desde A440
    tuning = librosa.estimate_tuning(y=y, sr=sr)

    # Calcular frecuencia fundamental
    frecuencia = 440 * 2**(tuning / 12)

    # Calcular factor de cambio de velocidad necesario para que suene a 440 Hz
    factor_velocidad = 2**(-tuning / 12)

    #   print(f"Desviación estimada: {tuning:.3f} semitonos")
    #   print(f"Frecuencia fundamental estimada: {frecuencia:.2f} Hz")
    #   print(f"Factor de velocidad necesario para corregir a 440 Hz: {factor_velocidad:.6f}")
    print(f"{frecuencia:.2f}")
    return frecuencia, factor_velocidad
def estimar_f0_con_pyin(audio_path):
    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=None)

    # Estimar f0 con pyin
    f0, voiced_flag, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=380,
        fmax=500,
        frame_length=2048,
        hop_length=256,
        resolution=0.01
    )

    # Filtrar los valores válidos
    f0_validas = f0[~np.isnan(f0)]

    if len(f0_validas) == 0:
        raise ValueError("No se detectaron frecuencias válidas con pyin.")

    # Usar la mediana para evitar outliers
    f0_median = np.median(f0_validas)
    factor_velocidad = 440 / f0_median

    #print(f"Frecuencia estimada con pyin: {f0_median:.2f} Hz")
    #print(f"Factor de velocidad necesario para afinar a 440 Hz: {factor_velocidad:.6f}")
    print(f"{f0_median:.2f}")
    return y, sr, f0_median, factor_velocidad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectar frecuencia fundamental y factor de velocidad.")
    parser.add_argument("audio_path", type=str, help="Ruta al archivo de audio")
    args = parser.parse_args()
    estimar_f0_con_pyin(args.audio_path)
