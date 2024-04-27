using System;
using System.Collections.Generic;
using System.IO;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.DiffSinger {

    [Phonemizer("DiffSinger Mandarin DIY RHY Phonemizer", "DIFFS CNM-DIY-RHY", "BaiTang", language: "DIFFS-RHY")]
    public class DiffSingerRhythmizerCHNDIYPhonemizer : DiffSingerRhythmizerBasePhonemizer {
        public Dictionary<string, string[]> chn2PhnDict = new Dictionary<string, string[]> { };
        public Dictionary<string, string[]> chn3PhnDict = new Dictionary<string, string[]> { };
        public Dictionary<string, string[]> jpnPhnDict = new Dictionary<string, string[]> { };

        public override void SetUpPhoneDictAndRhy() {
            LoadSingerRhythmizer("CNM");
            GetRealPhnDict("ds_CNM.txt");
            GetRhyMap();
            try {
                string path = Path.Combine(singer.Location, "ds_CNM.txt");
                this.chn2PhnDict = LoadDsDict(path);
            } catch (Exception) {
                return;
            }
            try {
                string path = Path.Combine(singer.Location, "ds_CNM3.txt");
                this.chn3PhnDict = LoadDsDict(path);
            } catch (Exception) {
                return;
            }
            try {
                string path = Path.Combine(singer.Location, "ds_JPN.txt");
                this.jpnPhnDict = LoadDsDict(path);
            } catch (Exception) {
                return;
            }
            AddDictKeySuffixToRealDict(chn2PhnDict, "&2");
            AddDictKeySuffixToRealDict(chn3PhnDict, "&3", "0");
            AddDictKeySuffixToRealDict(jpnPhnDict, "&j");
        }

        protected override void ProcessPart(Note[][] phrase) {
            ProcessByRhyMap(phrase);
        }

        protected override string[] Romanize(IEnumerable<string> lyrics) {
            return BaseChinesePhonemizer.Romanize(lyrics);
        }
    }
}
