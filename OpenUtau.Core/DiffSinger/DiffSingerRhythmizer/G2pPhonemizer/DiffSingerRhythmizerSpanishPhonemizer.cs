using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Spanish RHY Phonemizer", "DIFFS ES-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerSpanishPhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("ES");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_ES.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName() => "dsdict-es.yaml";
        protected override IG2p LoadBaseG2p() => new SpanishG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "a", "e", "i", "o", "u"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "b", "B", "ch", "d", "D", "f", "g", "G", "gn", "I", "k", "l", "ll", "m", "n", "p", "r", "rr", "s", "t", "U", "w", "x", "y", "Y", "z"
        };
    }
}