using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Portuguese RHY Phonemizer", "DIFFS PT-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerPortuguesePhonemizer : DiffSingerRhythmizerG2pPhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("PT");
            this.g2p = LoadG2p();
            GetRealPhnDict("ds_PT.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string GetDictionaryName() => "dsdict-pt.yaml";
        protected override IG2p LoadBaseG2p() => new PortugueseG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "E", "O", "a", "a~", "e", "e~", "i", "i~", "o", "o~", "u", "u~"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "J", "L", "R", "S", "X", "Z", "b", "d", "dZ", "f", "g", "j", "j~",
            "k", "l", "m", "n", "p", "r", "s", "t", "tS", "v", "w", "w~", "z"
        };
    }
}