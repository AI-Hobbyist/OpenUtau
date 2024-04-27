using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Russian RHY Phonemizer", "DIFFS RU-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerRussianPhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("RU");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_RU.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName()=>"dsdict-ru.yaml";
        protected override IG2p LoadBaseG2p() => new RussianG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "a", "aa", "ay", "ee", "i", "ii", "ja", "je", "jo", "ju", "oo",
            "u", "uj", "uu", "y", "yy"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "b", "bb", "c", "ch", "d", "dd", "f", "ff", "g", "gg", "h", "hh",
            "j", "k", "kk", "l", "ll", "m", "mm", "n", "nn", "p", "pp", "r", 
            "rr", "s", "sch", "sh", "ss", "t", "tt", "v", "vv", "z", "zh", "zz"
        };
    }
}