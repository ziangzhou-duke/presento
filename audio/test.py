from audio.stt import main
from audio.analyze_text import analyze

if __name__ == '__main__':
    res = main()
    print(analyze(res))
