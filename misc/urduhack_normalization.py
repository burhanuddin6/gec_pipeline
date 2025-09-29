'''
THIS CODE SNIPPET WAS TAKEN FROM URDUHACK PACKAGE, INSIDE THE `normalization` MODULE, BELONGING TO THE FILE `character.py`.
'''
from typing import Dict

# Contains wrong Urdu characters mapping to correct characters
CORRECT_URDU_CHARACTERS: Dict = {'آ': ['ﺁ', 'ﺂ'],
                                 'أ': ['ﺃ'],
                                 'ا': ['ﺍ', 'ﺎ', ],
                                 'ب': ['ﺏ', 'ﺐ', 'ﺑ', 'ﺒ'],
                                 'پ': ['ﭖ', 'ﭘ', 'ﭙ', ],
                                 'ت': ['ﺕ', 'ﺖ', 'ﺗ', 'ﺘ'],
                                 'ٹ': ['ﭦ', 'ﭧ', 'ﭨ', 'ﭩ'],
                                 'ث': ['ﺛ', 'ﺜ', 'ﺚ'],
                                 'ج': ['ﺝ', 'ﺞ', 'ﺟ', 'ﺠ'],
                                 'ح': ['ﺡ', 'ﺣ', 'ﺤ', 'ﺢ'],
                                 'خ': ['ﺧ', 'ﺨ', 'ﺦ'],
                                 'د': ['ﺩ', 'ﺪ'],
                                 'ذ': ['ﺬ', 'ﺫ'],
                                 'ر': ['ﺭ', 'ﺮ'],
                                 'ز': ['ﺯ', 'ﺰ', ],
                                 'س': ['ﺱ', 'ﺲ', 'ﺳ', 'ﺴ', ],
                                 'ش': ['ﺵ', 'ﺶ', 'ﺷ', 'ﺸ'],
                                 'ص': ['ﺹ', 'ﺺ', 'ﺻ', 'ﺼ', ],
                                 'ض': ['ﺽ', 'ﺾ', 'ﺿ', 'ﻀ'],
                                 'ط': ['ﻃ', 'ﻄ'],
                                 'ظ': ['ﻅ', 'ﻇ', 'ﻈ'],
                                 'ع': ['ﻉ', 'ﻊ', 'ﻋ', 'ﻌ', ],
                                 'غ': ['ﻍ', 'ﻏ', 'ﻐ', ],
                                 'ف': ['ﻑ', 'ﻒ', 'ﻓ', 'ﻔ', ],
                                 'ق': ['ﻕ', 'ﻖ', 'ﻗ', 'ﻘ', ],
                                 'ل': ['ﻝ', 'ﻞ', 'ﻟ', 'ﻠ', ],
                                 'م': ['ﻡ', 'ﻢ', 'ﻣ', 'ﻤ', ],
                                 'ن': ['ﻥ', 'ﻦ', 'ﻧ', 'ﻨ', ],
                                 'چ': ['ﭺ', 'ﭻ', 'ﭼ', 'ﭽ'],
                                 'ڈ': ['ﮈ', 'ﮉ'],
                                 'ڑ': ['ﮍ', 'ﮌ'],
                                 'ژ': ['ﮋ', ],
                                 'ک': ['ﮎ', 'ﮏ', 'ﮐ', 'ﮑ', 'ﻛ', 'ك'],
                                 'گ': ['ﮒ', 'ﮓ', 'ﮔ', 'ﮕ'],
                                 'ں': ['ﮞ', 'ﮟ'],
                                 'و': ['ﻮ', 'ﻭ', 'ﻮ', ],
                                 'ؤ': ['ﺅ'],
                                 'ھ': ['ﮪ', 'ﮬ', 'ﮭ', 'ﻬ', 'ﻫ', 'ﮫ'],
                                 'ہ': ['ﻩ', 'ﮦ', 'ﻪ', 'ﮧ', 'ﮩ', 'ﮨ', 'ه', ],
                                 'ۂ': [],
                                 'ۃ': ['ة'],
                                 'ء': ['ﺀ'],
                                 'ی': ['ﯼ', 'ى', 'ﯽ', 'ﻰ', 'ﻱ', 'ﻲ', 'ﯾ', 'ﯿ', 'ي'],
                                 'ئ': ['ﺋ', 'ﺌ', ],
                                 'ے': ['ﮮ', 'ﮯ', 'ﻳ', 'ﻴ', ],
                                 'ۓ': [],
                                 '۰': ['٠'],
                                 '۱': ['١'],
                                 '۲': ['٢'],
                                 '۳': ['٣'],
                                 '۴': ['٤'],
                                 '۵': ['٥'],
                                 '۶': ['٦'],
                                 '۷': ['٧'],
                                 '۸': ['٨'],
                                 '۹': ['٩'],
                                 '۔': [],
                                 '؟': [],
                                 '٫': [],
                                 '،': [],
                                 'لا': ['ﻻ', 'ﻼ'],
                                 '': ['ـ']

                                 }

_TRANSLATOR = {}
for key, value in CORRECT_URDU_CHARACTERS.items():
    _TRANSLATOR.update(dict.fromkeys(map(ord, value), key))


def normalize_characters(text: str) -> str:
    """
    The most important module in the UrduHack is the :py:mod:`~urduhack.normalization.character` module,
    defined in the module with the same name. You can use this module separately to normalize
    a piece of text to a proper specified Urdu range (0600-06FF). To get an understanding of how this module works, one
    needs to understand unicode. Every character has a unicode. You can search for any character unicode from any
    language you will find it. No two characters can have the same unicode. This module works with reference to the
    unicodes. Now as urdu language has its roots in Arabic, Parsian and Turkish. So we have to deal with all those
    characters and convert them to a normal urdu character. To get a bit more of what the above explanation means is.::

    >>> all_fes = ['ﻑ', 'ﻒ', 'ﻓ', 'ﻔ', ]
    >>> urdu_fe = 'ف'

    All the characters in all_fes are same but they come from different languages and they all have different unicodes.
    Now as computers deal with numbers, same character appearing in more than one place in a different language will
    have a different unicode and that will create confusion which will create problems in understanding the context of
    the data. :py:mod:`~character` module will eliminate this problem by replacing all the characters in all_fes by
    urdu_fe.

    This provides the functionality to replace wrong arabic characters with correct urdu characters and fixed the
    combine|join characters issue.

    Replace ``urdu`` text characters with correct ``unicode`` characters.

    Args:
        text (str): ``Urdu`` text
    Returns:
        str: Returns a ``str`` object containing normalized text.
    Examples:
        >>> from urduhack.normalization import normalize_characters
        >>> # Text containing characters from Arabic Unicode block
        >>> text = "مجھ کو جو توڑا ﮔیا تھا"
        >>> normalized_text = normalize_characters(text)
        >>> # Normalized text - Arabic characters are now replaced with Urdu characters
        >>> normalized_text
        مجھ کو جو توڑا گیا تھا
    """
    return text.translate(_TRANSLATOR)