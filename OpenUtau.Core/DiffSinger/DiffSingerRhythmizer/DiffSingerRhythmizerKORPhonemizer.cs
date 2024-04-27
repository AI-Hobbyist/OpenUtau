using System;
using System.Collections.Generic;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Korean RHY Phonemizer", "DIFFS KOR-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerKORPhonemizer : DiffSingerRhythmizerBasePhonemizer {
        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("KOR");
            GetRealPhnDict("ds_KOR.txt");
            GetRhyMap();
        }
        
        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        // Roman TODO
    }

}
