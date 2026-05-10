from src.config.config_loader import ConfigLoader
from src.tts.ttsable import TTSable
import src.utils as utils
import os
import re
import numpy as np
import soundfile as sf
import json
import requests
from subprocess import Popen, DEVNULL
import time
import sys
from src.tts.synthesization_options import SynthesizationOptions
from src.config.definitions.game_definitions import GameEnum

logger = utils.get_logger()


class TTSServiceFailure(Exception):
    pass

class VoiceModelNotFound(Exception):
    pass

class xVASynth(TTSable):
    """xVASynth TTS handler
    """
    @utils.time_it
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)
        self.__xvasynth_path = config.xvasynth_path
        self.__process_device = config.xvasynth_process_device
        self.__synthesize_url = 'http://127.0.0.1:8008/synthesize'
        self.__synthesize_batch_url = 'http://127.0.0.1:8008/synthesize_batch'
        self.__loadmodel_url = 'http://127.0.0.1:8008/loadModel'
        self.__setvocoder_url = 'http://127.0.0.1:8008/setVocoder'
        self.__model_path = f"{self.__xvasynth_path}/resources/app/models/{self._game.base_game.display_name}/"
        self.__pace = config.pace
        self.__use_sr = config.use_sr
        self.__use_cleanup = config.use_cleanup
        self.__model_type = ''
        self.__base_speaker_emb = ''
        if not self._facefx_path:
            self._facefx_path = self.__xvasynth_path + "/resources/app/plugins/lip_fuz"

        logger.log(self._loglevel, f'Connecting to xVASynth...')
        self._check_if_xvasynth_is_running()


    @utils.time_it
    def tts_synthesize(self, voiceline: str, final_voiceline_file: str, synth_options: SynthesizationOptions):
        phrases = self._split_voiceline(voiceline)
        voiceline_files = []
        for phrase in phrases:
            voiceline_file = f"{self._voiceline_folder}/{utils.clean_text(phrase)[:150]}.wav"
            voiceline_files.append(voiceline_file)

        if len(phrases) == 1:
            self._synthesize_line(phrases[0], final_voiceline_file, synth_options.aggro)
        else:
            # TODO: include batch synthesis for v3 models (batch not needed very often)
            if self.__model_type != 'xVAPitch':
                self._batch_synthesize(phrases, voiceline_files)
            else:
                for i, voiceline_file in enumerate(voiceline_files):
                    self._synthesize_line(phrases[i], voiceline_files[i])
            self._merge_audio_files(voiceline_files, final_voiceline_file)
    

    # Default fallback chains used when the NPC's CSV voice_model is missing
    # on disk. Order: most generic first. 0 = male, 1 = female in Skyrim's
    # actor base sex convention. Unknown gender → try both.
    _FALLBACK_CHAIN_MALE = ['MaleNord', 'MaleEvenToned', 'MaleSoldier']
    _FALLBACK_CHAIN_FEMALE = ['FemaleEvenToned']
    _FALLBACK_CHAIN_UNKNOWN = ['MaleNord', 'FemaleEvenToned', 'MaleEvenToned']

    def _resolve_voice_with_fallback(self, voice: str, xvasynth_acronym: str, voice_gender: int | None, nexus_link: str) -> str:
        """Returns the on-disk voice_path (without extension) for the requested
        voice model, or for the closest fallback by gender if the requested
        model is not installed. Raises VoiceModelNotFound if every candidate
        is missing.
        """
        def _path_for(model_name: str) -> str:
            return f"{self.__model_path}{xvasynth_acronym}{model_name.lower().replace(' ', '')}"

        primary_path = _path_for(voice)
        if os.path.exists(primary_path + '.json'):
            return primary_path

        if voice_gender == 0:
            chain = self._FALLBACK_CHAIN_MALE
        elif voice_gender == 1:
            chain = self._FALLBACK_CHAIN_FEMALE
        else:
            chain = self._FALLBACK_CHAIN_UNKNOWN

        for fallback_name in chain:
            fallback_path = _path_for(fallback_name)
            if os.path.exists(fallback_path + '.json'):
                logger.warning(
                    f"xVASynth voice model '{xvasynth_acronym}{voice.lower().replace(' ', '')}' not installed — "
                    f"using fallback '{xvasynth_acronym}{fallback_name.lower()}' (voice_gender={voice_gender})."
                )
                return fallback_path

        logger.error(
            f"Voice model does not exist in location '{primary_path}' and no fallback "
            f"({chain}) is installed either. Download the model from {nexus_link} "
            f"(Ctrl+F for '{xvasynth_acronym}{voice.lower().replace(' ', '')}')."
        )
        raise VoiceModelNotFound()

    @utils.time_it
    def change_voice(self, voice: str, in_game_voice: str | None = None, csv_in_game_voice: str | None = None, advanced_voice_model: str | None = None, voice_accent: str | None = None, voice_gender: int | None = None, voice_race: str | None = None):
        logger.log(self._loglevel, 'Loading voice model...')

        # this is a game check for Fallout4/Skyrim to correctly search the XVASynth voice models for the right game.
        if self._game.base_game == GameEnum.FALLOUT4:
            XVASynthAcronym="f4_"
            XVASynthModNexusLink="https://www.nexusmods.com/fallout4/mods/49340?tab=files"
        else:
            XVASynthAcronym="sk_"
            XVASynthModNexusLink = "https://www.nexusmods.com/skyrimspecialedition/mods/44184?tab=files"
        voice_path = self._resolve_voice_with_fallback(voice, XVASynthAcronym, voice_gender, XVASynthModNexusLink)

        with open(voice_path+'.json', 'r', encoding='utf-8') as f:
            voice_model_json = json.load(f)

        try:
            base_speaker_emb = voice_model_json['games'][0]['base_speaker_emb']
            base_speaker_emb = str(base_speaker_emb).replace('[','').replace(']','')
        except:
            base_speaker_emb = None

        self.__base_speaker_emb = base_speaker_emb
        self.__model_type = voice_model_json.get('modelType')
    
        model_change = {
            'outputs': None,
            'version': '3.0',
            'model': voice_path, 
            'modelType': self.__model_type,
            'base_lang': self._language, 
            'pluginsContext': '{}',
        }
        #For some reason older 1.0 model will load in a way where they only emit high pitched static noise about 20-30% of the time, this series of run_backupmodel calls below 
        #are here to prevent the static issues by loading the model by following a sequence of model versions of 
        # 3.0 -> 1.1  (will fail to load) -> 3.0 -> 1.1 -> make a dummy voice sample with _synthesize_line -> 1.0 (will fail to load) -> 3.0 -> 1.0 again
        if voice_model_json.get('modelVersion') == 1.0:
            logger.log(self._loglevel, '1.0 model detected running following sequence to bypass voice model issues : 3.0 -> 1.1  (will fail to load) -> 3.0 -> 1.1 -> make a dummy voice sample with _synthesize_line -> 1.0 (will fail to load) -> 3.0 -> 1.0 again')
            if self._game.base_game == GameEnum.FALLOUT4:
                backup_voice='piper'
                self._run_backup_model(backup_voice)
                backup_voice='maleeventoned'
                self._run_backup_model(backup_voice)
                backup_voice='piper'
                self._run_backup_model(backup_voice)
                backup_voice='maleeventoned'
                self._run_backup_model(backup_voice)
                self._synthesize_line("test phrase", f"{self._voiceline_folder}/temp.wav", False ,'1.0')
            else:
                backup_voice='malenord'
                self._run_backup_model(backup_voice)
        try:
            requests.post(self.__loadmodel_url, json=model_change)
            self._last_voice = voice
            logger.log(self._loglevel, f'Target model {voice} loaded.')
        except:
            logger.error(f'Target model {voice} failed to load.')
            #This step is vital to get older voice models (1,1 and lower) to run
            if self._game.base_game == GameEnum.FALLOUT4:
                backup_voice='piper'
            else:
                backup_voice='malenord'
            self._run_backup_model(backup_voice)
            try:
                requests.post(self.__loadmodel_url, json=model_change)
                self._last_voice = voice
                logger.log(self._loglevel, f'Voice model {voice} loaded.')
            except:
                logger.error(f'model {voice} failed to load. Try restarting Mantella')
                input('\nPress any key to stop Mantella...')
                sys.exit(0)
    

    @utils.time_it
    def _split_voiceline(self, voiceline, max_length=150):
        """Split voiceline into phrases by commas, 'and', and 'or'"""
        def group_sentences(voiceline_sentences, max_length=150):
            """
            Splits sentences into separate voicelines based on their length (max=max_length)
            Groups sentences if they can be done so without exceeding max_length
            """
            grouped_sentences = []
            temp_group = []
            for sentence in voiceline_sentences:
                if len(sentence) > max_length:
                    grouped_sentences.append(sentence)
                elif len(' '.join(temp_group + [sentence])) <= max_length:
                    temp_group.append(sentence)
                else:
                    grouped_sentences.append(' '.join(temp_group))
                    temp_group = [sentence]
            if temp_group:
                grouped_sentences.append(' '.join(temp_group))

            return grouped_sentences

        # Split by commas and "and" or "or"
        chunks = re.split(r'(, | and | or )', voiceline)
        # Join the delimiters back to their respective chunks
        chunks = [chunks[i] + (chunks[i+1] if i+1 < len(chunks) else '') for i in range(0, len(chunks), 2)]
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]

        result = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                if result and result[-1].endswith(' and'):
                    result[-1] = result[-1][:-4]
                    chunk = 'and ' + chunk.strip()
                elif result and result[-1].endswith(' or'):
                    result[-1] = result[-1][:-3]
                    chunk = 'or ' + chunk.strip()
                result.append(chunk.strip())
            else:
                # Split long chunks based on length
                words = chunk.split()
                current_line = words[0]
                for word in words[1:]:
                    if len(current_line + ' ' + word) <= max_length:
                        current_line += ' ' + word
                    else:
                        if current_line.endswith(' and'):
                            current_line = current_line[:-4]
                            word = 'and ' + word
                        if current_line.endswith(' or'):
                            current_line = current_line[:-3]
                            word = 'or ' + word
                        result.append(current_line.strip())
                        current_line = word
                result.append(current_line.strip())

        result = group_sentences(result, max_length)
        logger.debug(f'Split sentence into : {result}')

        return result
    

    @utils.time_it
    def _merge_audio_files(self, audio_files, voiceline_file_name):
        merged_audio = np.array([])

        for audio_file in audio_files:
            try:
                audio, samplerate = sf.read(audio_file)
                merged_audio = np.concatenate((merged_audio, audio))
                sf.write(voiceline_file_name, merged_audio, samplerate)
            except:
                logger.error(f'Could not find voiceline file: {audio_file}')


    @utils.time_it
    def _synthesize_line(self, line, save_path, aggro: bool = False, voicemodelversion='3.0'):
        pluginsContext = {}
        # in combat
        if aggro:
            pluginsContext["mantella_settings"] = {
                "emAngry": 0.6
            }
        data = {
            'pluginsContext': json.dumps(pluginsContext),
            'modelType': self.__model_type,
            'sequence': line,
            'pace': self.__pace,
            'outfile': save_path,
            'vocoder': 'n/a',
            'base_lang': self._language,
            'base_emb': self.__base_speaker_emb,
            'useSR': self.__use_sr,
            'useCleanup': self.__use_cleanup,
        }

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                requests.post(self.__synthesize_url, json=data)
                break  # exit the loop if the request is successful
            except ConnectionError as e:
                if attempt < max_attempts - 1:  # if not the last attempt
                    logger.warning(f"Connection error while synthesizing voiceline. Restarting xVASynth server... ({attempt})")
                    if voicemodelversion!='1.0':
                        self._run_xvasynth_server()
                        self.change_voice(self._last_voice)
                else:
                    logger.error(f"Failed to synthesize line after {max_attempts} attempts. Skipping voiceline: {line}")
                    break


    @utils.time_it
    def _batch_synthesize(self, grouped_sentences, voiceline_files):
        # line = [text, unknown 1, unknown 2, pace, output_path, unknown 5, unknown 6, pitch_amp]
        linesBatch = [[grouped_sentences[i], '', '', 1, voiceline_files[i], '', '', 1] for i in range(len(grouped_sentences))]
        
        data = {
            'pluginsContext': '{}',
            'modelType': self.__model_type,
            'linesBatch': linesBatch,
            'speaker_i': None,
            'vocoder': [],
            'outputJSON': None,
            'useSR': None,
            'useCleanup': None,
        }

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                requests.post(self.__synthesize_batch_url, json=data)
                break  # Exit the loop if the request is successful
            except ConnectionError as e:
                if attempt < max_attempts - 1:  # Not the last attempt
                    logger.warning(f"Connection error while synthesizing voiceline. Restarting xVASynth server... ({attempt})")
                    self._run_xvasynth_server()
                    self.change_voice(self._last_voice)
                else:
                    logger.error(f"Failed to synthesize line after {max_attempts} attempts. Skipping voiceline: {linesBatch}")
                    break


    @utils.time_it
    def _check_if_xvasynth_is_running(self):
        self._times_checked += 1
        try:
            if (self._times_checked > 15):
                # break loop
                logger.error(f'Could not connect to xVASynth after {self._times_checked} attempts. Ensure that xVASynth is running and restart Mantella.')
                raise TTSServiceFailure()

            # contact local xVASynth server; ~2 second timeout
            response = requests.get('http://127.0.0.1:8008/')
            response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
        except requests.exceptions.RequestException as err:
            if ('Connection aborted' in err.__str__()):
                # So it is alive
                return

            if (self._times_checked == 1):
                self._run_xvasynth_server()
            # do the web request again
            return self._check_if_xvasynth_is_running()
        

    @utils.time_it
    def _run_xvasynth_server(self):
        try:
            # start the process without waiting for a response
            if (self._tts_print):
                # print subprocess output
                Popen(f'{self.__xvasynth_path}/resources/app/cpython_{self.__process_device}/server.exe', cwd=self.__xvasynth_path, stdout=None, stderr=None)
            else:
                # ignore output
                Popen(f'{self.__xvasynth_path}/resources/app/cpython_{self.__process_device}/server.exe', cwd=self.__xvasynth_path, stdout=DEVNULL, stderr=DEVNULL)

            time.sleep(1)
        except:
            logger.error(f'Could not run xVASynth. Ensure that the path "{self.__xvasynth_path}" is correct.')
            raise TTSServiceFailure()
        

    @utils.time_it
    def _run_backup_model(self, voice):
        logger.log(self._loglevel, f'Attempting to load backup model {voice}.')
        #This function exists only to force XVASynth to play older models properly by resetting them by loading models in sequence
        
        #If for some reason the model fails to load (for example, because it's an older model) then Mantella will attempt to load a backup model. 
        #This will allow the older model to load without errors 
            
        if self._game.base_game == GameEnum.FALLOUT4:
            XVASynthAcronym="f4_"
            XVASynthModNexusLink="https://www.nexusmods.com/fallout4/mods/49340?tab=files"
            #voice='maleeventoned'
        else:
            XVASynthAcronym="sk_"
            XVASynthModNexusLink = "https://www.nexusmods.com/skyrimspecialedition/mods/44184?tab=files"
            #voice='malenord'
        voice_path = self._resolve_voice_with_fallback(voice, XVASynthAcronym, None, XVASynthModNexusLink)

        with open(voice_path+'.json', 'r', encoding='utf-8') as f:
            voice_model_json = json.load(f)

        try:
            base_speaker_emb = voice_model_json['games'][0]['base_speaker_emb']
            base_speaker_emb = str(base_speaker_emb).replace('[','').replace(']','')
        except:
            base_speaker_emb = None

        backup_model_type = voice_model_json.get('modelType')
        
        backup_model_change = {
            'outputs': None,
            'version': '3.0',
            'model': voice_path, 
            'modelType': backup_model_type,
            'base_lang': self._language, 
            'pluginsContext': '{}',
        }
        try:
            requests.post(self.__loadmodel_url, json=backup_model_change)
            logger.log(self._loglevel, f'Backup model {voice} loaded.')
        except:
            logger.error(f"Backup model {voice} failed to load")
