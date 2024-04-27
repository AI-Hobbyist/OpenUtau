using System.Collections.Generic;
using System.Linq;
using OpenUtau.Api;
using OpenUtau.Core.G2p;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger Cantonese Dsdur Phonemizer", "DIFFS CNY-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurCNYPhonemizer : DiffSingerDsdurBasePhonemizer {
        protected override string GetDictionaryName() => "dsdict-cny.yaml";
        protected override string GetDictionaryDsdurName() => "dsdur_cny";
        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return ZhG2p.CantoneseInstance.Convert(lyrics.ToList(), false, true).Split(" ");
        }
    }
}