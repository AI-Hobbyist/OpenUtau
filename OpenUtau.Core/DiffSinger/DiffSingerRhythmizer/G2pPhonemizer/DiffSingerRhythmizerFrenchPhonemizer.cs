using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger French RHY Phonemizer", "DIFFS FR-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerFrenchPhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("FR");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_FR.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName() => "dsdict-fr.yaml";
        protected override IG2p LoadBaseG2p() => new FrenchG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "aa", "ai", "an", "au", "ee", "ei", "eu", "ii", "in", "oe", "on", "oo", "ou", "un", "uu", "uy"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "bb", "ch", "dd", "ff", "gg", "gn", "jj", "kk", "ll", "mm", "nn", "pp", "rr", "ss", "tt", "vv", "ww", "yy", "zz"
        };
    }
}