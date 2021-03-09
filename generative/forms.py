from django import forms

choices = (
    ('',''),
    ('austria', 'Austria'), 
    ('czech', 'Czech'), 
    ('france', 'France',), 
    ('hungary', 'Hungary'), 
    ('yugoslavia', 'Yugoslavia'), 
    ('netherlands', 'Netherlands'), 
    ('poland', 'Poland'), 
    ('switzerland', 'Switzerland'), 
    ('natmin', 'Natmin'), 
    # ('germany_ballad', 'Germany (Ballad)'), 
    # ('germany_boehme', 'Germany (Boehme)'), 
    # ('germany_dva', 'Germany (Dva)'), 
    # ('germany_erk', 'Germany (Erk)'), 
    # ('germany_fink', 'Germany (Fink)'), 
    # ('germany_kinder', 'Germany (Kinder)'), 
    # ('germany_zuccal', 'Germany (Zuccal)'), 
    # ('china_han', 'China (Han)'), 
    # ('china_natmin', 'China (Natmin)'), 
    # ('china_shanxi', 'China (Shanxi)'), 
    # ('china_xinhua', 'China (Xinhua)'), 
)

class GenerateAudioForm(forms.Form):
    region = forms.ChoiceField(choices=choices, required=True)
    temp = forms.CharField(required=True)