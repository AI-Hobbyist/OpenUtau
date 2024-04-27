using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenUtau.Api;
using OpenUtau.Core.G2p;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Cantonese RHY Phonemizer", "DIFFS CNY-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerCNYPhonemizer : DiffSingerRhythmizerBasePhonemizer {

        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("CNY");
            GetRealPhnDict("ds_CNY.txt");
            GetRhyMap();
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return ZhG2p.CantoneseInstance.Convert(lyrics.ToList(), false, true).Split(" ");
        }
    }

}
