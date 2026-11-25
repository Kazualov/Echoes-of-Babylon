# Error analysis (validation, shared split)

Each system is analysed on its first seed. Length buckets are terciles of the source length in bytes; 'damaged' documents contain `<gap>`/`x` markers in the transliteration.


## Scores by subset and error statistics

| system | GM all (n=156) | GM short (n=52) | GM medium (n=52) | GM long (n=52) | GM damaged (n=68) | GM intact (n=88) | GM first 256 bytes | number recall | name recall | median length ratio | too short (<0.5) | empty | repetitive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| byt5-mbr | 17.19 | 37.59 | 17.82 | 13.64 | 14.86 | 19.09 | 33.09 | 0.561 | 0.492 | 1.19 | 4 | 0 | 94 |
| byt5-member2 | 16.35 | 33.51 | 17.96 | 12.95 | 14.38 | 17.95 | 28.57 | 0.475 | 0.439 | 0.99 | 22 | 0 | 84 |
| byt5 | 20.6 | 42.05 | 22.25 | 16.34 | 19.07 | 21.84 | 35.07 | 0.588 | 0.526 | 0.97 | 16 | 0 | 77 |
| mbart50 | 42.44 | 49.95 | 44.55 | 40.2 | 41.79 | 42.72 | 48.27 | 0.817 | 0.679 | 0.98 | 7 | 0 | 5 |
| mbr3-ensemble | 39.79 | 48.79 | 40.8 | 37.81 | 39.69 | 39.88 | 46.57 | 0.825 | 0.672 | 1.0 | 5 | 0 | 12 |
| qwen2.5-7b-lora | 34.98 | 41.62 | 37.32 | 32.77 | 35.49 | 34.56 | 41.14 | 0.777 | 0.627 | 0.93 | 6 | 0 | 3 |

- **GM first 256 bytes**: outputs and references both cut to 256 bytes before scoring (diagnostic only).
- **number recall**: share of numbers in the reference (amounts, weights) that appear in the output.
- **name recall**: share of capitalised words (mostly personal and place names) reproduced.
- **length ratio**: output words / reference words; < 0.5 usually means a truncated or stopped translation.


## 3 worst documents per system (sentence chrF++)

**byt5-mbr**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 0.7

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: Aššur-ṭāb son of Elamma seized us against Aššur-ṭāb son of Ali-ahum, and Aššur-ṭāb seized us against Aššur-ṭāb son of Ali-ahum, and Aššur-ṭāb said to Aššur-ṭāb son of Ali-ahum: The proceeds from 21 talents 9 minas of washed copper belonging to Aššur-ṭāb son of Elamma's son and I seized us against Aš

**byt5-mbr**, `507d179f-f4eb-40c6-8781-4ccb2050cad3`, chrF++ 7.8

- source: um-ma ša-lim-a-šùr-ma a-na a-šùr-iš-tí-kál en-na-nim en-um-a-šùr DUMU ú-zu-a ù en-um-a-šùr qí-bi-ma i-nu-mì 17 GÍN.TA i-zi-zu 15 GÍN.TA a-na-ku áš-a-am AN.NA za-ku a-ḫu-ú-a a-tù-nu i-na ša-ḫa-at ṣú-ḫa-ri-im i-zi-za-ma AN.NA ù TÚG.ḪI.A ki-ma KÙ.BABBAR 1 ma-na tù-šé-lá-a-ni-ni ep-ša-ma KÙ.BABBAR pá-ni
- reference: From Šalim-Aššur to Aššuriš-tikal, Ennānum, Ennam-Aššur son of Uzuwa and Ennam-Aššur: When the price (for tin) stood at 1:17 I bought at 1:15; the tin is cleared. My dear brothers, assist the servant and act for the tin and textiles in such a way that you realise every last mina of silver for me, an
- output: Thus Šalim-Aššur, say to Aššur-ištikal, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum,

**byt5-mbr**, `5df8b056-0e84-46d3-a4b9-411ee84b7501`, chrF++ 9.7

- source: a-na ku-li-a qí-bi-ma um-ma a-sà-num-ma a-na-kam i-qá-qá-ar ší-a-ma-tim wa-áš-ba-ku ú ší-mu-um ma-ad ší-ma-am ša ta-šé-e-ú i li-bi-kà ša ba-lá-ṭí-kà ú ba-lá-ṭí-ni a-na-kam-ma ú-šé-lá šál-ma-ku mì-ma i li-bi₄-kà lá ta-na-ḫi-id ší-ma-am šu-ra-am ša i-dí-ú i li-bi-i ma-tí-im ša ḫa-tim KÙ.BABBAR šé-bi-l
- reference: To Kuliya, thus Asānum: I am staying here in an area of purchases and there is much for sale! Purchases at which you are striving, which will secure you and us a profit, I can realize here. I am well, there is not reason to worry. <gap> purchase goods that they are in the countryside of hattum. Send
- output: To Kuliya from Asānum: As for these proceedings and the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds f

**byt5-member2**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 2.9

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: From Aššur-ṭāb son of Elamma to Aššur-ṭāb son of Ali-ahum and Aššur-ṭāb son of Ali-ahum seized us against Aššur-ṭāb son of Ali-ahum. Witnessed by Ikūn-pīya, by Uṣur-ša-Aššur. Witnessed by Ikūn-pīya, by Uṣur-ša-Aššur."

**byt5-member2**, `507d179f-f4eb-40c6-8781-4ccb2050cad3`, chrF++ 7.4

- source: um-ma ša-lim-a-šùr-ma a-na a-šùr-iš-tí-kál en-na-nim en-um-a-šùr DUMU ú-zu-a ù en-um-a-šùr qí-bi-ma i-nu-mì 17 GÍN.TA i-zi-zu 15 GÍN.TA a-na-ku áš-a-am AN.NA za-ku a-ḫu-ú-a a-tù-nu i-na ša-ḫa-at ṣú-ḫa-ri-im i-zi-za-ma AN.NA ù TÚG.ḪI.A ki-ma KÙ.BABBAR 1 ma-na tù-šé-lá-a-ni-ni ep-ša-ma KÙ.BABBAR pá-ni
- reference: From Šalim-Aššur to Aššuriš-tikal, Ennānum, Ennam-Aššur son of Uzuwa and Ennam-Aššur: When the price (for tin) stood at 1:17 I bought at 1:15; the tin is cleared. My dear brothers, assist the servant and act for the tin and textiles in such a way that you realise every last mina of silver for me, an
- output: Thus Šalim-Aššur, say to Aššur-ištikal, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum, Ennānum,

**byt5-member2**, `5df8b056-0e84-46d3-a4b9-411ee84b7501`, chrF++ 9.6

- source: a-na ku-li-a qí-bi-ma um-ma a-sà-num-ma a-na-kam i-qá-qá-ar ší-a-ma-tim wa-áš-ba-ku ú ší-mu-um ma-ad ší-ma-am ša ta-šé-e-ú i li-bi-kà ša ba-lá-ṭí-kà ú ba-lá-ṭí-ni a-na-kam-ma ú-šé-lá šál-ma-ku mì-ma i li-bi₄-kà lá ta-na-ḫi-id ší-ma-am šu-ra-am ša i-dí-ú i li-bi-i ma-tí-im ša ḫa-tim KÙ.BABBAR šé-bi-l
- reference: To Kuliya, thus Asānum: I am staying here in an area of purchases and there is much for sale! Purchases at which you are striving, which will secure you and us a profit, I can realize here. I am well, there is not reason to worry. <gap> purchase goods that they are in the countryside of hattum. Send
- output: To Kuliya from Asānum: As for these proceedings and the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds from the proceeds f

**byt5**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 0.7

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: Aššur-ṭāb son of Elamma seized us against Aššur-ṭāb son of Ali-ahum, and Aššur-ṭāb seized us against Aššur-ṭāb son of Ali-ahum, and Aššur-ṭāb said to Aššur-ṭāb son of Ali-ahum: The proceeds from 21 talents 9 minas of washed copper belonging to Aššur-ṭāb son of Elamma's son and I seized us against Aš

**byt5**, `8e7d176d-b52a-41cf-838b-b8a74be3a00d`, chrF++ 5.2

- source: KIŠIB IŠTAR-pá-li-il₅ DUMU šu-ku-bi₄-im KIŠIB ni-mar-IŠTAR DUMU ba-la KIŠIB a-bu-ša-lim DUMU ì-lí-a-lúm 0.5 ma-na KÙ.BABBAR ṣa-ru-pá-am i-ṣé-er IŠTAR-pá-li-il₅ DUMU šu-ku-bi₄-im DAM.GÀR-ru-um i-šu iš-tù ḫa-muš-tim ša {d}EN.LÍL-ba-ni ITU.KAM ṭí-i-na-tim 1 GÍN.TA i-na ITU.KAM a-na KÙ.BABBAR a-nim ṣí-i
- reference: Seal of Ištar-pālil, son of Šu-Kūbum, seal of Nimar-Ištar, son of Bala, seal of Abu-šalim, son of Ilī-ālum. 0.5 mina of refined silver Ištar-pālil, son of Šu-Kūbum, owes to the creditor. From the week-eponymy of Ellil-bāni, month X, he will add 1 shekel per month for this silver. Eponymy of hanna-nā
- output: Seal of Ištar-palil, son of Šu-Kūbum, seal of Nimar-Ištar, son of Bala, 

**byt5**, `3b1208c6-cfcc-4c5c-a8aa-73842cc91ff3`, chrF++ 5.8

- source: KIŠIB a-ta-tí-a DUMU en-na-nim KIŠIB a-ḫu-wa-qar DUMU a-šur-SIPA KIŠIB a-mur-a-šur DUMU šu-eš-ḫa-ra KIŠIB a-šur-{d}UTU DUMU ili₅-ba-ni ša a-al-DU₁₀ a-mur-a-šur a-na iṣ-ba-at-ni-a-tí-ma um-ma a-al-DU₁₀-ma a-na a-mur-a-šur-ma KÙ.BABBAR a-ta ù šu-IŠTAR qá-tám mi-ìš-lá i-na ša i-ip-ṭí-re ša-qá-lá-am lá 
- reference: Seal of Atatiya, son of Ennānum, seal of Ahu-waqar, son of Aššur-rē'ī, seal of Amur-Aššur, son of Šu-Išhara, seal of Aššur-šamšī, son of Ilī-bāni. Āl-ṭāb seized us against Amur-Aššur and Āl-ṭāb said to Amur-Aššur: Have you and Šu-Ištar not promised (or: been ordered) to pay each half of the silver?"
- output: Seal of Atatiya, son of Ennānum,

**mbart50**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 4.1

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: Lullu, Ennam-Aššur and Abum-šalim, representing Aššur-ṭāb son of Elamma, seized us against Aššur-ṭāb son of Ali-ahum and said: The proceeds from the 21 talents 9 minas of washed copper belonging to Aššur-ṭāb son of Elamma that was deposited in your house you gave him for purchases." Aššur-ṭāb answer

**mbart50**, `8597f793-fd5a-4b97-a812-b4d292031227`, chrF++ 12.1

- source: i-na ša-pá-at na-ri-im iš-tí-in e-me-ra-am a-qí-i iš-tí-in i-na ur-šu-ú a-na IŠTAR be-lá-at kà-le-ta a-qí-i iš-tí-in i-na ma-ma ILLAT-tum iq-be-ma a-qí-i 15 GÍN AN.NA a-na e-ri-qí-im áš-qúl
- reference: On the river-bank I sacrificed one sheep;
- output: I have been paid in the name of the gentleman with whom I met. I have been paid in Uršu to Ištar, the slave of Kaletta. He has been paid in accordance with the regulation of the caravan and I have been paid. I paid 15 shekels of tin for the donkeys.

**mbart50**, `d9638cf9-a743-4b24-b8de-ab82740ae75a`, chrF++ 12.3

- source: 1.3333 ma-na 6 GÍN KÙ.BABBAR i pá-ni-tim ḫa-ta-a-ni ù e-wa-ri-mu-ša ub-lu-nim a-ḫa-ma 1 ma-na 19 GÍN KÙ.BABBAR i ša-ni-tim šé-pí-šu-nu 1 ma-na KÙ.BABBAR ší-im sú-ub-ri-im pá-ni-tim ù wa-ar-ki-ú-tim ša ir-dí-ú-ni-ni i-dí-nu-nim ŠÀ.BA 5.25 GÍN KÙ.BABBAR iš-tí nu-ki-ri-bi-im a-ḫu-a-šu 1.5 ma-na KÙ.BABB
- reference: Hatāni and Ewarimuša brought previously 1.3333 mina 6 shekels of silver; further, 1 mina 19 shekels of silver in their second transport. They gave 1 mina of silver, the price of the first slave and the earlier ones which they conducted here. Thereof: 5 1 / 4 shekels of silver owed by the gardener Ah
- output: 1.3333 mina 6 shekels of silver hatānu and ewarimuša brought to us in the first instance.

**mbr3-ensemble**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 4.1

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: Lullu, Ennam-Aššur and Abum-šalim, representing Aššur-ṭāb son of Elamma, seized us against Aššur-ṭāb son of Ali-ahum and said: The proceeds from the 21 talents 9 minas of washed copper belonging to Aššur-ṭāb son of Elamma that was deposited in your house you gave him for purchases." Aššur-ṭāb answer

**mbr3-ensemble**, `8597f793-fd5a-4b97-a812-b4d292031227`, chrF++ 12.1

- source: i-na ša-pá-at na-ri-im iš-tí-in e-me-ra-am a-qí-i iš-tí-in i-na ur-šu-ú a-na IŠTAR be-lá-at kà-le-ta a-qí-i iš-tí-in i-na ma-ma ILLAT-tum iq-be-ma a-qí-i 15 GÍN AN.NA a-na e-ri-qí-im áš-qúl
- reference: On the river-bank I sacrificed one sheep;
- output: I have been paid in the name of the gentleman with whom I met. I have been paid in Uršu to Ištar, the slave of Kaletta. He has been paid in accordance with the regulation of the caravan and I have been paid. I paid 15 shekels of tin for the donkeys.

**mbr3-ensemble**, `76c16d4a-432f-4dfc-8c94-aa3399bd3ee6`, chrF++ 15.4

- source: 1 ma-na URUDU a-na da-di-a a-šu-mì il₅-ba-ni áš-qúl <gap> ma-na a-na il₅-ba-ni a-dí-in <gap> ma-na 5 GÍN ší-im a-ri-im <gap> ma-na i-na <gap> mu-ṣí-šu áš-qúl 1 ma-na <gap> šu-ša-lu ŠÀ.BA 3.6666 ma-na 5 GÍN ḫa-bu-lá-ak-šu-um
- reference: 1 mina of copper I paid to Dadiya in the name of Ilī-bāni; I gave <gap> mina(s) to Ilī-bāni; <gap> mina 5 shekels as payment for a hide; <gap> mina(s) in <gap>, his exit fee, I paid; 1 mina <gap> Thereof I am indebted to him for 3 minas 45 shekels.
- output: I paid 1 mina of copper to Dadiya for Ilī-bāni.

**qwen2.5-7b-lora**, `08f7b5b5-8a52-43af-a30a-ad9c11ce8de7`, chrF++ 2.8

- source: lu-lu en-nam-a-šur ù a-bu-um-ša-lim ša ki-ma a-šùr-DU₁₀ DUMU e-lá-ma a-na a-šur-DU₁₀ DUMU a-lá-ḫi-im iṣ-bu-tù-ni-a-tí-ma um-ma šu-nu-ma ší-im 21 GÚ 9 ma-na URUDU ma-sí-im ša a-šur-DU₁₀ DUMU e-lá-ma ša i-na É-tí-kà i-ni-id-ú a-na ší-mì-im ta-dí-šu um-ma a-šur-DU₁₀-ma a-dí-šu-ma 17.8333 ma-na 2 GÍN KÙ
- reference: Lullu,
- output: Lulu, Ennam-Aššur and Abu-šalim seized us against Aššur-ṭāb son of Elamma, saying: Give us the proceeds from the 21 talents 9 minas of washed copper that Aššur-ṭāb son of Elamma left with you." Aššur-ṭāb answered: "I shall send you 17.8333 minas 2 shekels of silver representing its price." The plena

**qwen2.5-7b-lora**, `fe95294d-1ef5-49aa-b6a9-5062bb2678a9`, chrF++ 13.1

- source: 2 ší-ta sà-ma-lá-tim ra-bi₄-a-tim a-<gap> ša ba-áb ni <gap> a-dí-šu-um 0.5 GÍN <gap> 4 sà-ma-lá-tim <gap> lá i-ḫa-li-qú <gap>
- reference: 2 big -cups <gap> of the gate <gap> I gave to him. ½ shekel<gap> 4 -cups <gap> should not be lost.
- output: 2 -textiles, the wages of the attorney, I shall send to you from the City.

**qwen2.5-7b-lora**, `76c16d4a-432f-4dfc-8c94-aa3399bd3ee6`, chrF++ 15.4

- source: 1 ma-na URUDU a-na da-di-a a-šu-mì il₅-ba-ni áš-qúl <gap> ma-na a-na il₅-ba-ni a-dí-in <gap> ma-na 5 GÍN ší-im a-ri-im <gap> ma-na i-na <gap> mu-ṣí-šu áš-qúl 1 ma-na <gap> šu-ša-lu ŠÀ.BA 3.6666 ma-na 5 GÍN ḫa-bu-lá-ak-šu-um
- reference: 1 mina of copper I paid to Dadiya in the name of Ilī-bāni; I gave <gap> mina(s) to Ilī-bāni; <gap> mina 5 shekels as payment for a hide; <gap> mina(s) in <gap>, his exit fee, I paid; 1 mina <gap> Thereof I am indebted to him for 3 minas 45 shekels.
- output: I paid 1 mina of copper to Dadiya for Ilī-bāni.

