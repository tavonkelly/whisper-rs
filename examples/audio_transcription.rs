// This example is not going to build in this folder.
// You need to copy this code into your project and add the dependencies whisper_rs and hound in your cargo.toml

use hound;
use std::fs::File;
use std::io::Write;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Loads a context and model, processes an audio file, and prints the resulting transcript to stdout.
fn main() -> Result<(), &'static str> {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify path to model as argument 1");
    let wav_path = std::env::args()
        .nth(2)
        .expect("Please specify path to wav file as argument 2");

    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::BaseEn,
    };

    // Enable DTW token level timestamp for unknown model by providing custom aheads
    // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    let custom_aheads = [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ]
    .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
        n_text_layer,
        n_head,
    });
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
        aheads: &custom_aheads,
    };

    // Load a context and model
    let ctx =
        WhisperContext::new_with_params(&model_path, context_param).expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create state");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(1);
    // Enable translation.
    params.set_translate(true);
    // Set the language to translate to to English.
    params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Open the audio file.
    let reader = hound::WavReader::open(wav_path).expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.expect("Invalid sample"))
        .collect();
    let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    // Convert audio to 16KHz mono f32 samples, as required by the model.
    // These utilities are provided for convenience, but can be replaced with custom conversion logic.
    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio).expect("Conversion error");
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }

    // Run the model.
    state.full(params, &audio[..]).expect("failed to run model");

    // Create a file to write the transcript to.
    let mut file = File::create("transcript.txt").expect("failed to create file");

    // Iterate through the segments of the transcript.
    for segment in state.as_iter() {
        // Get the transcribed text and timestamps for the current segment.
        let start_timestamp = segment.start_timestamp();
        let end_timestamp = segment.end_timestamp();

        let first_token_dtw_ts = if segment.n_tokens() > 0 {
            if let Some(token) = segment.get_token(0) {
                token.token_data().t_dtw
            } else {
                -1i64
            }
        } else {
            -1i64
        };
        // Print the segment to stdout.
        println!(
            "[{} - {} ({})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    Ok(())
}
