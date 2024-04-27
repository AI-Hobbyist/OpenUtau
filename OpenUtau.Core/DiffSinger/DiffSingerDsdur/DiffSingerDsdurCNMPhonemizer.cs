using System.Collections.Generic;

using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger Mandarin Dsdur Phonemizer", "DIFFS CNM-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurCNMPhonemizer : DiffSingerDsdurBasePhonemizer {
        protected override string GetDictionaryDsdurName() => "dsdur_cnm";
        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseChinesePhonemizer.Romanize(lyrics);
        }
    }
}