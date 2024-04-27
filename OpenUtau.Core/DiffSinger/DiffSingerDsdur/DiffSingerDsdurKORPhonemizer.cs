using System.Collections.Generic;

using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger Korean Dsdur Phonemizer", "DIFFS KOR-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurKORPhonemizer : DiffSingerDsdurBasePhonemizer {
        protected override string GetDictionaryDsdurName() => "dsdur_ko";
        protected override string GetDictionaryName() => "dsdict-ko.yaml";
    }
}