using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Italian RHY Phonemizer", "DIFFS IT-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerItalianPhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("IT");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_IT.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName() => "dsdict-it.yaml";
        protected override IG2p LoadBaseG2p() => new ItalianG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "a", "a1", "e", "e1", "EE", "i", "i1", "o", "o1", "OO", "u", "u1"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "b", "d", "dz", "dZZ", "f", "g", "JJ", "k", "l", "LL", "m", "n", "nf", "ng", "p", "r", "rr", "s", "SS", "t", "ts", "tSS", "v", "w", "y", "z"
        };
    }
}