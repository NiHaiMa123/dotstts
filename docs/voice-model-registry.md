# Local voice model registry

The local WebUI consumes accepted voice deployments. Training, checkpoint selection,
prompt selection, and post-processing tuning remain agent-run workflows; the WebUI
does not train or tune a model.

## Registry

`configs/voices/registry.yaml` defines the default model and the ordered list shown
by the WebUI. Every entry points to one strict voice profile. The server validates
all profiles and their bound artifact checksums before it opens the browser.

A profile binds every setting that can change the generated voice or final audio:

- official base model path and revision;
- trainable-delta/LoRA directory, selected training step, weights hash, and metadata
  hash;
- prompt WAV, its hash, and the exact prompt transcript;
- runtime precision, optimization mode, sequence limits, vocoder merge setting, and
  LoRA merge policy;
- language, template, text normalization, speaker scale, ODE method, step count,
  guidance, seed, sentence pause, and text split size;
- edge-trim and voice-polish configurations with file hashes;
- model-specific TXT input and final WAV output directories.

The selected profile is resolved once when a job starts. The state machine records
its version and canonical SHA-256. Switching models releases the previously loaded
runtime before loading the new base/adapter pair, so CUDA memory and settings cannot
leak from one voice to another.

## Adding a newly accepted LoRA

After the agent has finished training and post-processing review:

1. Freeze the chosen checkpoint, prompt WAV/transcript, inference parameters, and
   post-processing YAML. Do not rewrite historical Slice acceptance files.
2. Copy `configs/voices/fuxuan_step400_v1.yaml` to a new uniquely named profile and
   replace every model-specific field and checksum.
3. Give the model its own input and output directories so identically named TXT files
   cannot overwrite another voice's WAV files. Convention: TXT files for a model go
   in `inputs/<model>_text/` and generated WAVs land in `outputs/<model>_audio/`.
   Create both folders when the profile is created; the WebUI also creates any
   missing profile directories at startup as a safety net.
4. Add the profile path to `configs/voices/registry.yaml`; change
   `default_model_id` only when the new model has been explicitly accepted as the
   default.
5. Run `python -m unittest tests.test_voice_registry tests.test_fuxuan_webui` and
   then the complete test suite.
6. Restart the WebUI. The dropdown is populated from the registry automatically;
   HTML and Python source changes are not required.

If any artifact is missing or its bytes differ from the recorded hash, startup or
model loading fails explicitly instead of silently mixing a LoRA with another
voice's prompt or post-processing chain.
