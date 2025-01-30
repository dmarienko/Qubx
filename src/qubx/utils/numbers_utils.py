def count_decimal_places(number: float) -> int:
    number_str = ("%.10f" % number).strip("0")
    if "." in number_str:
        integer_part, decimal_part = number_str.split(".")
        return len(decimal_part)
    else:
        return 0
