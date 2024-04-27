using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using Serilog;

namespace OpenUtau.Core.DiffSinger {

    public abstract class DiffSingerRhythmizerG2pPhonemizer : DiffSingerRhythmizerBasePhonemizer {
        public IG2p g2p;
        protected virtual string GetDictionaryName() => "dsdict.yaml";
        protected virtual IG2p LoadBaseG2p() => new ArpabetG2p();
        //vowels and consonants of BaseG2p
        protected virtual string[] GetBaseG2pVowels() => new string[] { };
        protected virtual string[] GetBaseG2pConsonants() => new string[] { };

        protected virtual IG2p LoadG2p() {
            string rootPath;
            if (File.Exists(Path.Join(singer.Location, "dsdur", "dsconfig.yaml"))) {
                rootPath = Path.Combine(singer.Location, "dsdur");
            } else {
                rootPath = singer.Location;
            }
            var dictionaryName = GetDictionaryName();
            var g2ps = new List<IG2p>();
            // Load dictionary from plugin folder.
            string path = Path.Combine(PluginDir, dictionaryName);
            if (File.Exists(path)) {
                try {
                    g2ps.Add(G2pDictionary.NewBuilder().Load(File.ReadAllText(path)).Build());
                } catch (Exception e) {
                    Log.Error(e, $"Failed to load {path}");
                }
            }

            // Load dictionary from singer folder.
            var replacements = new Dictionary<string, string>();
            string file = Path.Combine(rootPath, dictionaryName);
            if (File.Exists(file)) {
                try {
                    g2ps.Add(G2pDictionary.NewBuilder().Load(File.ReadAllText(file)).Build());
                    replacements = G2pReplacementsData.Load(File.ReadAllText(file)).toDict();
                } catch (Exception e) {
                    Log.Error(e, $"Failed to load {file}");
                }
            }

            // Load base g2p.
            var baseG2p = LoadBaseG2p();
            if (baseG2p == null) {
                return new G2pFallbacks(g2ps.ToArray());
            }
            var phonemeSymbols = new Dictionary<string, bool>();
            foreach (var v in GetBaseG2pVowels()) {
                phonemeSymbols[v] = true;
            }
            foreach (var c in GetBaseG2pConsonants()) {
                phonemeSymbols[c] = false;
            }
            foreach (var from in replacements.Keys) {
                var to = replacements[from];
                if (baseG2p.IsValidSymbol(to)) {
                    if (baseG2p.IsVowel(to)) {
                        phonemeSymbols[from] = true;
                    } else {
                        phonemeSymbols[from] = false;
                    }
                }
            }
            g2ps.Add(new G2pRemapper(baseG2p, phonemeSymbols, replacements));
            return new G2pFallbacks(g2ps.ToArray());
        }

        public string[] GetSymbols(Note note) {
            //priority:
            //1. phonetic hint
            //2. query from g2p dictionary
            //3. treat lyric as phonetic hint, including single phoneme
            //4. default pause
            if (!string.IsNullOrEmpty(note.phoneticHint)) {
                // Split space-separated symbols into an array.
                return note.phoneticHint.Split()
                    .Where(s => g2p.IsValidSymbol(s)) // skip the invalid symbols.
                    .ToArray();
            }
            // User has not provided hint, query g2p dictionary.
            var g2presult = g2p.Query(note.lyric)
                ?? g2p.Query(note.lyric.ToLowerInvariant());
            if (g2presult != null) {
                return g2presult;
            }
            //not founded in g2p dictionary, treat lyric as phonetic hint
            var lyricSplited = note.lyric.Split()
                    .Where(s => g2p.IsValidSymbol(s)) // skip the invalid symbols.
                    .ToArray();
            if (lyricSplited.Length > 0) {
                return lyricSplited;
            }
            return new string[] { "SP" };
        }

        public string[] GetRealPhnFromSymbols(string[] symbols) {
            var realPhnList = new List<string> { };
            foreach (var syb in symbols) {
                if (this.realPhnDict.ContainsKey(syb)) {
                    realPhnList.AddRange(this.realPhnDict[syb].ToList());
                } else {
                    realPhnList.Add(syb);
                }
            }
            return realPhnList.ToArray();
        }

        //TODO polysyllabic word support
        public override void ProcessByRhyMap(Note[][] phrase) {
            float padding = 0.5f;//Padding time for consonants at the beginning of a sentence
            var phonemes = new List<string> { "SP" };
            var realPhonemes = new List<string> { "SP" };
            var midi = new List<long> { 0 };//Phoneme pitch
            var midi_dur = new List<float> { padding };//List of parent note duration for each phoneme
            var is_slur = new List<bool> { false };//Whether the phoneme is slur
            List<double> ph_dur;//Phoneme durations output by the model
            var notePhIndex = new List<int> { 1 };//The position of the first phoneme of each note in the phoneme list
            var phAlignPoints = new List<Tuple<int, double>>();//Phoneme alignment position, s, absolute time
            double offsetMs = timeAxis.TickPosToMsPos(phrase[0][0].position);

            //Convert note list to phoneme list
            foreach (int groupIndex in Enumerable.Range(0, phrase.Length)) {
                string[] notePhonemes;
                string[] realNotePhonemes;
                Note[] group = phrase[groupIndex];
                if (group[0].phoneticHint is null) {
                    // var lyric = group[0].lyric;
                    // if (realPhnDict.ContainsKey(lyric)) {
                    //     realNotePhonemes = realPhnDict[lyric];
                    //     notePhonemes = GetSimilarPhonemes(realNotePhonemes);
                    // } else {
                    //     realNotePhonemes = new string[] { lyric };
                    //     notePhonemes = new string[] { lyric };
                    // }
                    realNotePhonemes = GetRealPhnFromSymbols(GetSymbols(group[0]));
                    notePhonemes = GetSimilarPhonemes(realNotePhonemes);
                } else {
                    realNotePhonemes = group[0].phoneticHint.Split(" ");
                    notePhonemes = GetSimilarPhonemes(realNotePhonemes);
                }
                is_slur.AddRange(Enumerable.Repeat(false, notePhonemes.Length));
                phAlignPoints.Add(new Tuple<int, double>(
                    phonemes.Count + (notePhonemes.Length > 1 ? 1 : 0),//TODO
                    timeAxis.TickPosToMsPos(group[0].position) / 1000
                    ));
                phonemes.AddRange(notePhonemes);
                realPhonemes.AddRange(realNotePhonemes);
                midi.AddRange(Enumerable.Repeat((long)group[0].tone, notePhonemes.Length));
                notePhIndex.Add(phonemes.Count);

                midi_dur.AddRange(Enumerable.Repeat((float)timeAxis.MsBetweenTickPos(
                    group[0].position, group[^1].position + group[^1].duration) / 1000, notePhonemes.Length));
            }
            var lastNote = phrase[^1][^1];
            phAlignPoints.Add(new Tuple<int, double>(
                phonemes.Count,
                timeAxis.TickPosToMsPos(lastNote.position + lastNote.duration) / 1000));

            long defaultToken = rhythmizer.phonemes.IndexOf("SP");
            var tokens = phonemes
                .Select(x => (long)(rhythmizer.phonemes.IndexOf(x)))
                .Select(x => x < 0 ? defaultToken : x)
                .ToList();
            var inputs = new List<NamedOnnxValue>();
            inputs.Add(NamedOnnxValue.CreateFromTensor("tokens",
                new DenseTensor<long>(tokens.ToArray(), new int[] { tokens.Count }, false)
                .Reshape(new int[] { 1, tokens.Count })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("midi",
                new DenseTensor<long>(midi.ToArray(), new int[] { midi.Count }, false)
                .Reshape(new int[] { 1, midi.Count })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("midi_dur",
                new DenseTensor<float>(midi_dur.ToArray(), new int[] { midi_dur.Count }, false)
                .Reshape(new int[] { 1, midi_dur.Count })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("is_slur",
                new DenseTensor<bool>(is_slur.ToArray(), new int[] { is_slur.Count }, false)
                .Reshape(new int[] { 1, is_slur.Count })));
            var outputs = rhythmizer.session.Run(inputs);
            ph_dur = outputs.First().AsTensor<float>().Select(x => (double)x).ToList();
            var positions = new List<double>();
            List<double> alignGroup = ph_dur.GetRange(0, phAlignPoints[0].Item1);
            positions.AddRange(stretch(alignGroup, 1, phAlignPoints[0].Item2));
            foreach (var pair in phAlignPoints.Zip(phAlignPoints.Skip(1), (a, b) => Tuple.Create(a, b))) {
                var currAlignPoint = pair.Item1;
                var nextAlignPoint = pair.Item2;
                alignGroup = ph_dur.GetRange(currAlignPoint.Item1, nextAlignPoint.Item1 - currAlignPoint.Item1);
                double ratio = (nextAlignPoint.Item2 - currAlignPoint.Item2) / alignGroup.Sum();
                positions.AddRange(stretch(alignGroup, ratio, nextAlignPoint.Item2));
            }
            int index = 1;
            foreach (int groupIndex in Enumerable.Range(0, phrase.Length)) {
                Note[] group = phrase[groupIndex];
                var noteResult = new List<Tuple<string, int>>();
                if (group[0].lyric.StartsWith("+")) {
                    continue;
                }
                double notePos = timeAxis.TickPosToMsPos(group[0].position);//音符起点位置，单位ms
                for (int phIndex = notePhIndex[groupIndex]; phIndex < notePhIndex[groupIndex + 1]; ++phIndex) {
                    noteResult.Add(Tuple.Create(realPhonemes[phIndex], timeAxis.TicksBetweenMsPos(
                       notePos, positions[phIndex] * 1000)));
                }
                partResult[group[0].position] = noteResult;
            }
        }
    }
}
