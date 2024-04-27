using System;
using System.Collections.Generic;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger English RHY Phonemizer(deprecated)", "DIFFS ENG-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerENGPhonemizer : DiffSingerRhythmizerBasePhonemizer {

        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("ENG");
            GetRealPhnDict("ds_ENG.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }
    }
}
