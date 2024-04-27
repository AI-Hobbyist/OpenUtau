using System.Collections.Generic;
using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Mandarin RHY Phonemizer", "DIFFS CNM-RHY", "BaiTang",language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerCHNPhonemizer : DiffSingerRhythmizerBasePhonemizer {
        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseChinesePhonemizer.Romanize(lyrics);
        }
    }

}
