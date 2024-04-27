**English** | [简体中文](README_zh.md)

# OpenUTAU for DiffSinger Multi-Language
# 版本特点
## 1. 音素器
### 1.1 DiffSinger Rhythmizer Phonemizer系列音素器
加入了几个DiffSinger Rhythmizer Phonemizer，可以将其他语种或其他派系的音素映射到Opencpop的Rhythmizer上，并使用其预测音素时长。
歌手目录下需要准备`rhy_map.txt`音素映射文件`ds_XXX.txt`语种字典：`ds_CNM|ds_CNM3.txt|ds_CNY|ds_JPN|ds_ENG|ds_KOR|ds_ARPA|ds_FR|ds_DE|ds_IT|ds_ES|ds_PT|ds_RU`，其中英语ARPA音素器为主推使用,CNM3为中文三段式音素器。内置有g2p的语种（英语、法语、德语、意大利语、西班牙语、葡萄牙语、俄语），词典只需声明音标到音素的对应，无需声明单词到音素的对应。
### rhy_map.txt
```
k1[需映射音素],k[对应rhy里面的音素]
```

### 示例ds_XXX.txt
```
ka,k a
kaa,k a a
br,AP
```

注：
1. DiffSinger Mandarin Rhythmizer Phonemizer 和 DiffSinger Mandarin DIY Rhythmizer Phonemizer的区别是：前者只能用Opencpop里面的音素表和rhy，后者可以使用自定义的中文字典`ds_CNM.txt`。
2. 在两个有自定义词典的中文音素器中增加歌词后缀换词典解析功能。&2用二段中文解析，&3用三段中文解析，&j用日语解析。

### 1.2 DiffSinger Dsdur Phonemizer系列音素器
dsdur系列音素器适配多语种方案。dsdur可以放在歌手目录下使用，也可以放在dep目录下供无dsdur的歌手使用。dsdur系列音素器可以自定义每个语种使用的dsdur预测模型，为在Dependencies目录下的子目录：
`dsdur_cnm | dsdur_cnm3 | dsdur_cny | dsdur_jp | dsdur_en | dsdur_ko`,
所以目前音素器识别dsdur位置的顺序为：歌手目录下的dsdur，Dependencies下的dsdur_xx目录，Dependencies下的dsdur目录

注：
1. 这里提供colstone的multi-lang五语种词典对应的词典配置，包括Rhy和Dsdur系列音素器的，https://github.com/atonyxu/Multi-langs_Dictionary
2. DiffSinger Mandarin Rhythmizer/Dsdur Phonemizer 和 DiffSinger Mandarin DIY Rhythmizer/Dsdur Phonemizer，可直接输入汉字，会自动转拼音。
3. DiffSinger Cantonese Rhythmizer/Dsdur Phonemizer，可直接输入汉字（简体），会自动转粤语拼音。
4. DiffSinger Japanese Rhythmizer/Dsdur Phonemizer，可直接输入假名，会自动转换罗马音，字典里面无需再申明假名转换。

## 2. 批量功能
1. 自动添加AP，SP音素
2. 汉字转换粤语拼音
3. 字母大小写转换
4. 将被-分隔开的单词进行连接（openutau不支持-连接单词分配元音）

# OpenUtau
## Download
If you have already installed OpenUtau, you don't need to install again. Just download [nsf_hifigan vocoder](https://github.com/xunmengshe/OpenUtau/releases/0.0.0.0), launch OpenUtau, and drag the vocoder file into OpenUtau window to import.

For Windows users, please go to [Release Page](https://github.com/xunmengshe/OpenUtau/releases) and download the file with "diffsingerpack" which is shipped with vocoder.

For MacOS and Linux users, please download from [here](https://github.com/stakira/OpenUtau/releases), then download [nsf_hifigan vocoder](https://github.com/xunmengshe/OpenUtau/releases/0.0.0.0). Launch OpenUtau, and drag the vocoder file into OpenUtau window to import.

## About the state of this repo
This repo was the development repo of OpenUtau's diffsinger renderer. Now the diffsinger renderer has been merged to [the official repo of OpenUtau](https://github.com/stakira/openutau). These two repos are basically the same thing now.

This repo will continue to exist and provide diffsingerpack release for windows. Note that the official openutau and this repo provide the same functionalities. Both repos support diffsinger voicebanks and UTAU voicebanks. The only difference is that diffsingerpack ships with a vocoder so that you don't need to download it separately.

## Usage
[wiki/Usage](https://github.com/xunmengshe/OpenUtau/wiki/Usage)

## Send Feedback
If you run into any issue, feel free to provide your feedback:
- [Provide Feedback on GitHub](https://github.com/xunmengshe/OpenUtau/issues/new?assignees=&labels=&template=bug-report.yml)
- [Join DiffSinger Discord channel](https://discord.gg/JtKYyZgmGt)

It's suggested to provide a full snapshot of your OpenUtau window, your ustx file and OpenUtau log file.

## Links
- [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
- [DiffSinger (maintained by OpenVPI)](https://github.com/openvpi/DiffSinger)
- [DiffSinger Chinese Documentation](https://openvpi-docs.feishu.cn/wiki/space/7251035979191140356?ccm_open_type=lark_wiki_spaceLink)
- [OpenUtau official repo](https://github.com/stakira/OpenUtau)

---

Below is the README inherited from the original repository.

# OpenUtau

OpenUtau is a free, open-source editor made for the UTAU community.

[![AppVeyor](https://img.shields.io/appveyor/build/stakira/OpenUtau?style=for-the-badge&label=appveyor&logo=appveyor)](https://ci.appveyor.com/project/stakira/openutau)
[![Discord](https://img.shields.io/discord/551606189386104834?style=for-the-badge&label=discord&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/UfpMnqMmEM)
[![QQ Qroup](https://img.shields.io/badge/QQ-485658015-blue?style=for-the-badge)](https://qm.qq.com/cgi-bin/qm/qr?k=8EtEpehB1a-nfTNAnngTVqX3o9xoIxmT&jump_from=webapi)
[![Trello](https://img.shields.io/badge/trello-go-blue?style=for-the-badge&logo=trello)](https://trello.com/b/93ANoCIV/openutau)

## Getting started

[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=windows-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-win-x64.zip)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=windows-x86&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-win-x86.zip)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=macos-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-osx-x64.dmg)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=linux-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-linux-x64.tar.gz)

It is **strongly recommended** that you read these Github wiki pages before using the software.
- [Getting-Started](https://github.com/stakira/OpenUtau/wiki/Getting-Started)
- [Resamplers](https://github.com/stakira/OpenUtau/wiki/Resamplers-and-Wavtools)
- [Phonemizers](https://github.com/stakira/OpenUtau/wiki/Phonemizers)
- [FAQ](https://github.com/stakira/OpenUtau/wiki/FAQ)

- [中文使用说明](https://opensynth.miraheze.org/wiki/OpenUTAU/%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)

## How to contribute

Tried OpenUtau and not satisfied? Don't just walk away! You can help:
- Report issues on our [Discord server](https://discord.gg/UfpMnqMmEM) or Github.
- Suggest features on Discord or Github.
- Add or update translations for your language on Github.

Know how to code? Got an idea for an improvement? Don't keep it to yourself!
- Contribute fixes via pull requests.
- Check out the development roadmap on [Trello](https://trello.com/b/93ANoCIV/openutau) and discuss it on Discord.

## Plugin development

Want to contribute plugins to help other users? Check out our API documentation:
- [Editing Macros API Document](OpenUtau.Core/Editing/README.md)
- [Phonemizers API Document](OpenUtau.Core/Api/README.md)

## Main features

Navigate the interface naturally and fluently using the mouse and scroll wheel. Keyboard shortcuts are also available.

![Editor](Misc/GIFs/editor.gif)

Easily create songs and covers using the feature-rich MIDI editor.

![Editor](Misc/GIFs/editor2.gif)

Create expressive vibratos with the easy-to-use vibrato editor.

![Vibrato](Misc/GIFs/vibrato.gif)

Pre-rendering and built-in resamplers let you quickly preview your work.

![Playback](Misc/GIFs/playback.gif)

See the [Getting-Started Wiki page](https://github.com/stakira/OpenUtau/wiki/Getting-Started) for more!

## All features
- Modern user experience.
- Easy navigation using the mouse and keyboard.
- Feature-rich MIDI editor.
  - Support for importing VSQX (Vocaloid 4) tracks.
- Selective backward compatibility with UTAU.
  - OpenUtau aims to solve problems with fewer steps. It is not designed to replicate UTAU features exactly.
- Extensible real-time phonetic editing.
  - Includes phonemizers for different phonetic systems (VCV, CVVC, Arpasing, etc.) in many different languages (English, Japanese, Chinese, Korean, Russian and more).
- Expressions replace the standard UTAU "flags" for tuning.
  - The built-in WORLDLINE-R resampler supports curve tuning, similar to many vocal synth editors.
- Internationalisation, including UI translation and file system encoding support.
  - Unlike UTAU, there is no need to change your system locale to use OpenUtau.
- Smooth preview/rendering experience.
  - Pre-rendering allows OpenUtau to render vocals before playback, saving time during editing and tuning.
- Supports ENUNU AI singers. See the ![ENUNU wiki page](https://github.com/stakira/OpenUtau/wiki/Status-of-ENUNU-NNSVS-Support) for more info.
- Easy-to-use plugin system.
- Versatile resampling engine interface.
  - Compatible with most UTAU resamplers.
- Runs on Windows (32/64 bit), macOS, and Linux.

### What it doesn't do
- While OpenUtau can do very minimal mixing, it will not replace your digital audio workstation of choice.
- OpenUtau does not aim for Vocaloid compatibility, except for some limited features.
