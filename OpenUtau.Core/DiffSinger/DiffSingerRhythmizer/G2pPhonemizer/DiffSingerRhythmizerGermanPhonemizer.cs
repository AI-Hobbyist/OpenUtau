using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger German RHY Phonemizer", "DIFFS DE-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerGermanPhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("DE");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_DE.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName() => "dsdict-de.yaml";
        protected override IG2p LoadBaseG2p() => new GermanG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "ee", "eh", "er", "ex", "ih", "iy", "oe", "ohh", "ooh", "oy", "ue", "uh", "uw", "yy"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "b", "cc", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "pf", "q", "r", "rr", "s", "sh", "t", "th", "ts", "v", "w", "x", "y", "z", "zh"
        };
    }
}