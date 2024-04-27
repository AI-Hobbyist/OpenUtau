using System.Collections.Generic;

using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger Japanese Dsdur Phonemizer", "DIFFS JPN-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurJPNPhonemizer : DiffSingerDsdurBasePhonemizer {
        protected override string GetDictionaryName() => "dsdict-jp.yaml";
        protected override string GetDictionaryDsdurName() => "dsdur_jp";
        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseJapanesePhonemizer.Romanize(lyrics);
        }
    }
}