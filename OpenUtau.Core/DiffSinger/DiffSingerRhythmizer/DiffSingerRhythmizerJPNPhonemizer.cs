using System;
using System.Collections.Generic;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Japanese RHY Phonemizer", "DIFFS JPN-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerJPNPhonemizer : DiffSingerRhythmizerBasePhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("JPN");
            GetRealPhnDict("ds_JPN.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseJapanesePhonemizer.Romanize(lyrics);
        }
    }
}
