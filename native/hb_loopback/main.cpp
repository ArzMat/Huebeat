// hb_loopback - cattura l'audio di un processo (e figli) con l'API Windows
// Process Loopback e scrive PCM grezzo float32 stereo interleaved su stdout.
//
// Basato sul sample MIT "ApplicationLoopback" di Microsoft:
// https://github.com/microsoft/Windows-classic-samples/tree/main/Samples/ApplicationLoopback
//
// Uso: hb_loopback.exe <pid> [--exclude-tree] [--rate 48000]
//   default: include-tree (il processo e i suoi figli), 48000 Hz
//
// Output su stdout (binario puro, nessun header): frame interleaved L,R float32.
// Diagnostica su stderr.
//
// Exit code:
//   0 = stdout chiuso dal padre (uscita pulita)
//   1 = argomenti invalidi
//   2 = attivazione o inizializzazione audio fallita
//   3 = processo target terminato

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>

#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

// Handler di completamento per ActivateAudioInterfaceAsync.
// Il sample usa WRL (RuntimeClass + FtmBase); qui implementiamo a mano
// IUnknown + IAgileObject (marker, nessun metodo) per rendere l'oggetto
// agile tra apartment COM senza dipendere da WRL.
class ActivateHandler : public IActivateAudioInterfaceCompletionHandler,
                        public IAgileObject
{
public:
    ActivateHandler()
    {
        m_done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }

    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override
    {
        if (ppv == nullptr)
        {
            return E_POINTER;
        }
        if (riid == __uuidof(IUnknown) ||
            riid == __uuidof(IActivateAudioInterfaceCompletionHandler))
        {
            *ppv = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
        }
        else if (riid == __uuidof(IAgileObject))
        {
            *ppv = static_cast<IAgileObject*>(this);
        }
        else
        {
            *ppv = nullptr;
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    STDMETHODIMP_(ULONG) AddRef() override
    {
        return InterlockedIncrement(&m_ref);
    }

    STDMETHODIMP_(ULONG) Release() override
    {
        ULONG ref = InterlockedDecrement(&m_ref);
        if (ref == 0)
        {
            delete this;
        }
        return ref;
    }

    // IActivateAudioInterfaceCompletionHandler
    // Chiamato su un thread MTA quando l'attivazione e' completa.
    STDMETHODIMP ActivateCompleted(IActivateAudioInterfaceAsyncOperation* operation) override
    {
        HRESULT hrActivate = E_UNEXPECTED;
        IUnknown* punk = nullptr;
        HRESULT hr = operation->GetActivateResult(&hrActivate, &punk);
        if (FAILED(hr))
        {
            m_result = hr;
        }
        else if (FAILED(hrActivate))
        {
            m_result = hrActivate;
        }
        else if (punk == nullptr)
        {
            m_result = E_UNEXPECTED;
        }
        else
        {
            m_result = punk->QueryInterface(__uuidof(IAudioClient),
                                            reinterpret_cast<void**>(&m_client));
        }
        if (punk != nullptr)
        {
            punk->Release();
        }
        SetEvent(m_done);
        return S_OK;
    }

    HANDLE m_done = nullptr;          // segnalato a fine attivazione
    HRESULT m_result = E_UNEXPECTED;  // esito dell'attivazione
    IAudioClient* m_client = nullptr; // client audio ottenuto (se successo)

private:
    virtual ~ActivateHandler()
    {
        if (m_done != nullptr)
        {
            CloseHandle(m_done);
        }
    }

    volatile LONG m_ref = 1;
};

static void usage()
{
    fprintf(stderr,
            "uso: hb_loopback.exe <pid> [--exclude-tree] [--rate 48000]\n"
            "  cattura l'audio del processo <pid> (e figli) e scrive su stdout\n"
            "  PCM grezzo float32 stereo interleaved al rate richiesto.\n"
            "  --exclude-tree: cattura tutto TRANNE il processo e i suoi figli\n");
    fflush(stderr);
}

int wmain(int argc, wchar_t* argv[])
{
    // --- Parsing argomenti ---
    if (argc < 2)
    {
        usage();
        return 1;
    }

    wchar_t* end = nullptr;
    DWORD pid = wcstoul(argv[1], &end, 10);
    if (pid == 0 || end == nullptr || *end != L'\0')
    {
        usage();
        return 1;
    }

    bool includeTree = true;
    DWORD rate = 48000;
    for (int i = 2; i < argc; ++i)
    {
        if (wcscmp(argv[i], L"--exclude-tree") == 0)
        {
            includeTree = false;
        }
        else if (wcscmp(argv[i], L"--rate") == 0)
        {
            if (i + 1 >= argc)
            {
                usage();
                return 1;
            }
            ++i;
            rate = wcstoul(argv[i], &end, 10);
            if (rate < 8000 || rate > 192000 || end == nullptr || *end != L'\0')
            {
                usage();
                return 1;
            }
        }
        else
        {
            usage();
            return 1;
        }
    }

    // stdout in modalita' binaria: niente traduzione LF -> CRLF sui byte PCM
    if (_setmode(_fileno(stdout), _O_BINARY) == -1)
    {
        fprintf(stderr, "hb_loopback: impossibile mettere stdout in modalita' binaria\n");
        return 2;
    }

    // Handle del processo target: serve per accorgersi quando termina.
    HANDLE hProcess = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (hProcess == nullptr)
    {
        fprintf(stderr, "hb_loopback: OpenProcess(%lu) fallita, errore %lu (processo inesistente o accesso negato)\n",
                (unsigned long)pid, (unsigned long)GetLastError());
        return 2;
    }
    if (WaitForSingleObject(hProcess, 0) == WAIT_OBJECT_0)
    {
        fprintf(stderr, "hb_loopback: il processo %lu e' gia' terminato\n", (unsigned long)pid);
        return 3;
    }

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: CoInitializeEx fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    // --- Attivazione del dispositivo virtuale di process loopback ---
    // Parametri identici al sample Microsoft: ActivateAudioInterfaceAsync su
    // VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK con AUDIOCLIENT_ACTIVATION_PARAMS.
    AUDIOCLIENT_ACTIVATION_PARAMS activationParams = {};
    activationParams.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    activationParams.ProcessLoopbackParams.ProcessLoopbackMode = includeTree
        ? PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
        : PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;
    activationParams.ProcessLoopbackParams.TargetProcessId = pid;

    PROPVARIANT activateParams = {};
    activateParams.vt = VT_BLOB;
    activateParams.blob.cbSize = sizeof(activationParams);
    activateParams.blob.pBlobData = reinterpret_cast<BYTE*>(&activationParams);

    // Ref iniziale nostro; ActivateAudioInterfaceAsync fa AddRef/Release suoi.
    // L'oggetto vive fino a fine processo, nessuna Release finale necessaria.
    ActivateHandler* handler = new ActivateHandler();
    if (handler->m_done == nullptr)
    {
        fprintf(stderr, "hb_loopback: CreateEvent fallita\n");
        return 2;
    }

    IActivateAudioInterfaceAsyncOperation* asyncOp = nullptr;
    hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                                     __uuidof(IAudioClient),
                                     &activateParams,
                                     handler,
                                     &asyncOp);
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: ActivateAudioInterfaceAsync fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    if (WaitForSingleObject(handler->m_done, 10000) != WAIT_OBJECT_0)
    {
        fprintf(stderr, "hb_loopback: timeout in attesa dell'attivazione\n");
        return 2;
    }
    asyncOp->Release();

    if (FAILED(handler->m_result) || handler->m_client == nullptr)
    {
        fprintf(stderr, "hb_loopback: attivazione fallita, hr=0x%08lx\n", (unsigned long)handler->m_result);
        return 2;
    }
    IAudioClient* client = handler->m_client;

    // --- Formato imposto: float32, 2 canali, rate richiesto ---
    // Il dispositivo virtuale di process loopback non ha GetMixFormat: il
    // formato va dichiarato dal chiamante (come nel sample). AUTOCONVERTPCM
    // + SRC fanno convertire il mix engine verso il formato richiesto.
    WAVEFORMATEX fmt = {};
    fmt.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    fmt.nChannels = 2;
    fmt.nSamplesPerSec = rate;
    fmt.wBitsPerSample = 32;
    fmt.nBlockAlign = static_cast<WORD>(fmt.nChannels * fmt.wBitsPerSample / 8);
    fmt.nAvgBytesPerSec = fmt.nSamplesPerSec * fmt.nBlockAlign;
    fmt.cbSize = 0;

    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                            AUDCLNT_STREAMFLAGS_LOOPBACK |
                            AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                            AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                            0,
                            0,
                            &fmt,
                            nullptr);
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: IAudioClient::Initialize fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    HANDLE hSampleReady = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (hSampleReady == nullptr)
    {
        fprintf(stderr, "hb_loopback: CreateEvent fallita\n");
        return 2;
    }
    hr = client->SetEventHandle(hSampleReady);
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: SetEventHandle fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    IAudioCaptureClient* capture = nullptr;
    hr = client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&capture));
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: GetService(IAudioCaptureClient) fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    hr = client->Start();
    if (FAILED(hr))
    {
        fprintf(stderr, "hb_loopback: IAudioClient::Start fallita, hr=0x%08lx\n", (unsigned long)hr);
        return 2;
    }

    fprintf(stderr, "hb_loopback: cattura avviata, pid=%lu mode=%s formato=float32 stereo %lu Hz\n",
            (unsigned long)pid, includeTree ? "include-tree" : "exclude-tree", (unsigned long)rate);
    fflush(stderr);

    // --- Loop di cattura ---
    // Attende l'evento sample-ready O la morte del processo target; il
    // timeout tiene vivo il loop anche se il target non produce audio.
    // Come nel sample: a ogni risveglio si drenano TUTTI i pacchetti
    // disponibili (GetNextPacketSize in loop), perche' l'engine puo'
    // accumularne piu' d'uno tra un evento e l'altro.
    HANDLE waits[2] = { hSampleReady, hProcess };
    std::vector<BYTE> zeros; // buffer di zeri per i pacchetti SILENT

    for (;;)
    {
        DWORD w = WaitForMultipleObjects(2, waits, FALSE, 500);
        if (w == WAIT_OBJECT_0 + 1)
        {
            fprintf(stderr, "hb_loopback: processo target %lu terminato, esco\n", (unsigned long)pid);
            client->Stop();
            return 3;
        }
        if (w == WAIT_FAILED)
        {
            fprintf(stderr, "hb_loopback: WaitForMultipleObjects fallita, errore %lu\n",
                    (unsigned long)GetLastError());
            return 2;
        }
        // w == WAIT_OBJECT_0 oppure WAIT_TIMEOUT: drena i pacchetti disponibili

        UINT32 framesAvailable = 0;
        while (SUCCEEDED(capture->GetNextPacketSize(&framesAvailable)) && framesAvailable > 0)
        {
            BYTE* data = nullptr;
            DWORD flags = 0;
            hr = capture->GetBuffer(&data, &framesAvailable, &flags, nullptr, nullptr);
            if (FAILED(hr))
            {
                fprintf(stderr, "hb_loopback: GetBuffer fallita, hr=0x%08lx\n", (unsigned long)hr);
                return 2;
            }

            size_t cb = static_cast<size_t>(framesAvailable) * fmt.nBlockAlign;
            const BYTE* src = data;
            if ((flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0)
            {
                // Pacchetto silenzioso: scrive zeri della stessa lunghezza,
                // mai saltare, il lettore si aspetta un flusso continuo.
                if (zeros.size() < cb)
                {
                    zeros.assign(cb, 0);
                }
                src = zeros.data();
            }

            size_t written = fwrite(src, 1, cb, stdout);
            capture->ReleaseBuffer(framesAvailable);

            if (written != cb || fflush(stdout) != 0)
            {
                // stdout chiuso (il padre e' morto): uscita pulita
                fprintf(stderr, "hb_loopback: stdout chiuso, esco\n");
                client->Stop();
                return 0;
            }
        }
    }
}
