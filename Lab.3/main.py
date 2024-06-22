import os
import wfdb
import matplotlib.pyplot as plt

def plot_ctg_signal(signal, title, fs):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Czas (s)')
    plt.ylabel('Wartość sygnału')
    plt.show()

def browse_ctg_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.hea'):
                file_path = os.path.join(root, file)
                record_name = file_path[:-4]
                try:
                    record = wfdb.rdrecord(record_name)
                    signals = record.p_signal
                    fs = record.fs

                    for i, signal in enumerate(signals.T):
                        plot_ctg_signal(signal, f'Sygnał {i + 1} z pliku {file}', fs)
                except Exception as e:
                    print(f"Nie udało się odczytać pliku {file_path}: {e}")

ctg_directory_path = './mit-bih-ecg-compression-test-database-1.0.0'
browse_ctg_directory(ctg_directory_path)
