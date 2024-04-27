using System.Collections.Generic;

using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger Mandarin 三段式 Dsdur Phonemizer", "DIFFS CNM-TRI-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurCNMTRIPhonemizer : DiffSingerDsdurBasePhonemizer {
        protected override string GetDictionaryName() => "dsdict-cnm3.yaml";
        protected override string GetDictionaryDsdurName() => "dsdur_cnm3";
        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseChinesePhonemizer.Romanize(lyrics);
        }
    }
}