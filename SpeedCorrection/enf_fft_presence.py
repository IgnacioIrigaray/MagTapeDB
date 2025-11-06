import librosa
import numpy as np
import argparse

def detectar_enf_fft(audio_path, umbral_presencia=10):
    y, sr = librosa.load(audio_path, sr=None)

    # Aplicar ventana
    y_win = y * np.hanning(len(y))

    # FFT
    espectro = np.fft.rfft(y_win)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    magnitudes = np.abs(espectro)

    # Banda 40-60 Hz
    banda_mask = (freqs >= 40) & (freqs <= 60)
    freqs_banda = freqs[banda_mask]
    mags_banda = magnitudes[banda_mask]

    if len(freqs_banda) == 0:
        raise ValueError("No hay datos en la banda 40-60 Hz.")

    # Pico y amplitud media
    idx_pico = np.argmax(mags_banda)
    enf_pico = freqs_banda[idx_pico]
    amp_pico = mags_banda[idx_pico]
    amp_media = np.mean(mags_banda)

    presencia = amp_pico / amp_media
    hay_enf = presencia >= umbral_presencia

    return enf_pico, amp_media, presencia, hay_enf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectar ENF en banda 40-60 Hz con indicador de amplitud relativa.")
    parser.add_argument("audio_path", type=str, help="Ruta al archivo de audio")
    parser.add_argument("--presencia_minima", type=float, default=15.0, help="Relación pico/media mínima para presencia ENF")
    parser.add_argument("--batch", action="store_true", help="Modo batch: salida en CSV")
    args = parser.parse_args()

    try:
        enf_pico, amp_media, presencia, hay_enf = detectar_enf_fft(
            args.audio_path, umbral_presencia=args.presencia_minima
        )

        if args.batch:
            print(f"{enf_pico:.3f},{amp_media:.6f},{presencia:.3f},{int(hay_enf)}")
        else:
            print(f"ENF estimada (pico): {enf_pico:.3f} Hz")
            print(f"Amplitud media en banda: {amp_media:.6f}")
            print(f"Relación pico/media: {presencia:.3f}")
            print(f"¿ENF presente? {'Sí' if hay_enf else 'No'}")

    except Exception as e:
        if args.batch:
            print("NaN,NaN,NaN,0")
        else:
            print(f"Error: {e}")
