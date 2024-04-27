using System.Collections.Generic;
using OpenUtau.Api;
using OpenUtau.Core.G2p;

namespace OpenUtau.Core.DiffSinger {
    [Phonemizer("DiffSinger English-ARPA Dsdur Phonemizer", "DIFFS ENG-DSDUR", "BaiTang", language: "DIFFS-DSDUR")]
    public class DiffSingerDsdurENGPhonemizer : DiffSingerDsdurG2pPhonemizer {
        protected override string GetDictionaryName() => "dsdict-en.txt";
        protected override string GetDictionaryDsdurName() => "dsdur_en";
        protected override IG2p LoadBaseG2p() => new ArpabetG2p();
        protected override string[] GetBaseG2pVowels() => new string[] {
            "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er",
            "ey", "ih", "iy", "ow", "oy", "uh", "uw"
        };

        protected override string[] GetBaseG2pConsonants() => new string[] {
            "b", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n",
            "ng", "p", "r", "s", "sh", "t", "th", "v", "w", "y", "z", "zh"
        };
    }
}