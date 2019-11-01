import unicodedata
import os

name_to_chr = {}

for i in range(ord("א"), ord("ת") + 1):
    letter = chr(i)
    name = unicodedata.name(letter)
    name = name.replace("HEBREW LETTER ", "")
    parts = name.split(" ")

    if len(parts) == 1:
        main_name = parts[0].title()
        name_to_chr[main_name] = letter

    else:
        main_name = parts[1].title()
        name_to_chr["{}-{}".format(main_name, "final")] = letter

# replace some keys manually as the training classes provided are not named consistently
name_to_chr['Mem-medial'] = name_to_chr.pop('Mem')
name_to_chr['Mem'] = name_to_chr.pop('Mem-final')

name_to_chr['Nun-medial'] = name_to_chr.pop('Nun')

name_to_chr['Tsadi-medial'] = name_to_chr.pop('Tsadi')

# replace some different spelling
name_to_chr['Taw'] = name_to_chr.pop('Tav')
name_to_chr['Waw'] = name_to_chr.pop('Vav')

chr_to_name = {name_to_chr[key]: key for key in name_to_chr}


# a = [ [ ["a", "b", "c"], ["a1", "b1", "c1"]  ],  [ ["d", "e", "f"], ["d1", "e1", "f1"]  ]  ]

def save_document(document_text, name, folder):
    if not os.path.exists("{}/{}".format(folder, "output")):
        os.makedirs("{}/{}".format(folder, "output"))

    with open("{}/{}/{}.txt".format(folder, "output", name), "w", encoding="utf-8") as text_file:
        text_file.write(document_text)


def convert_to_text(document):
    """
        The document parameter is a 3 level list representing a document, lines and words.

        There is NO need to perform reversal on characters or words as python recognises
        that hebrew in RTL and prints the characters in the correct direction.

        The characters in each word are joined.
        The words in each sentence are joined with a space.
        The sentences are joined using a new lines to create the document.

    """

    # join character in words together, join words in sentence with space, join sentences
    # in document with new line
    document_text = "\n".join(
        [" ".join(
            ["".join(list(map(lambda x: name_to_chr[x], word))) for word in line]) for line in document])
    return document_text


"""
  OLD COMMENT OLD COMMENT!!!
  The document parameter is a 3 level list representing a document, lines and words.

  The character in each word are reversed to reflect rtl order and then joined.
  The words in each sentence are reversed to reflect rtl order and joined with a space.
  The sentences are joined using a new lines to create the document.



  # document_text = "\n".join(
  #   [" ".join(
  #      ["".join(list(map(lambda x: name_to_chr[x], word))[::-1]) for word in line][::-1]) for line in document])

  new_doc = []
  for line in document:
      new_line = []

      # reverse order of characters in words
      for word in line:
          new_word = word#[::-1]
          new_line.append(list(map(lambda x: name_to_chr[x], new_word)))

      # reverse order of words in sentence
      new_line = new_line#[::-1]
      new_doc.append(new_line)

  """