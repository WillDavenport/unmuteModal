import pytest

from unmute.llm.llm_utils import rechunk_to_words, rechunk_to_sentences


async def make_iterator(s: str):
    parts = s.split("|")
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_rechunk_to_words():
    test_strings = [
        "hel|lo| |w|orld",
        "hello world",
        "hello \nworld",
        "hello| |world",
        "hello| |world|.",
        "h|e|l|l|o| |\tw|o|r|l|d|.",
        "h|e|l|l|o\n| |w|o|r|l|d|.",
    ]

    for s in test_strings:
        parts = [x async for x in rechunk_to_words(make_iterator(s))]
        assert parts[0] == "hello"
        assert parts[1] == " world" or parts[1] == " world."

    async def f(s: str):
        x = [x async for x in rechunk_to_words(make_iterator(s))]
        print(x)
        return x

    assert await f("i am ok") == ["i", " am", " ok"]
    assert await f(" i am ok") == [" i", " am", " ok"]
    assert await f(" they are ok") == [" they", " are", " ok"]
    assert await f("  foo bar") == [" foo", " bar"]
    assert await f(" \t foo  bar") == [" foo", " bar"]


@pytest.mark.asyncio
async def test_rechunk_to_sentences():
    """Test sentence chunking functionality"""
    
    async def f(s: str):
        x = [x async for x in rechunk_to_sentences(make_iterator(s))]
        print(f"Input: {s!r} -> Output: {x}")
        return x

    # Basic sentence tests
    assert await f("Hello world.") == ["Hello world."]
    assert await f("Hello world. How are you?") == ["Hello world.", "How are you?"]
    assert await f("Hello world! How are you? I'm fine.") == ["Hello world!", "How are you?", "I'm fine."]
    
    # Test streaming input (simulating LLM token-by-token output)
    assert await f("Hel|lo wo|rld. H|ow a|re y|ou?") == ["Hello world.", "How are you?"]
    assert await f("This| is| a| test.| How| are| you?") == ["This is a test.", "How are you?"]
    
    # Test abbreviations (should not split)
    assert await f("Dr. Smith went to the store.") == ["Dr. Smith went to the store."]
    assert await f("Mr. Johnson vs. Mr. Brown.") == ["Mr. Johnson vs. Mr. Brown."]
    assert await f("The company Inc. was founded.") == ["The company Inc. was founded."]
    
    # Test decimal numbers (should not split)
    assert await f("The price is 3.14 dollars.") == ["The price is 3.14 dollars."]
    assert await f("Pi equals 3.14159 approximately.") == ["Pi equals 3.14159 approximately."]
    
    # Test multiple sentence endings
    assert await f("What?! Really?? Yes!!!") == ["What?!", "Really??", "Yes!!!"]
    
    # Test incomplete sentences (should yield remaining text)
    assert await f("This is incomplete") == ["This is incomplete"]
    assert await f("First sentence. This is incomplete") == ["First sentence.", "This is incomplete"]
    
    # Test whitespace handling
    assert await f("Hello.   How are you?") == ["Hello.", "How are you?"]
    assert await f("Hello.\n\nHow are you?") == ["Hello.", "How are you?"]
    
    # Test empty input
    assert await f("") == []
    assert await f("   ") == []
    
    # Test edge cases with punctuation
    assert await f("Hello... world.") == ["Hello... world."]
    assert await f("Are you sure??? Yes!") == ["Are you sure???", "Yes!"]
