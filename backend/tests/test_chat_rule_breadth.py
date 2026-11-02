import pytest

from backend.services.conversation_analysis import asks_sensitive_information, is_payment_request

PAYMENT_POSITIVE = [
    "Please pay the courier charges before we ship your laptop.",
    "You must transfer the insurance deposit by evening.",
    "Send a documentation fee of 500 to this account.",
    "Remit the activation cost and your account goes live.",
    "The training contribution has to be paid before the first class.",
    "Go and buy gift cards worth 200 and read me the codes.",
    "Purchase two vouchers from the shop and share the numbers.",
    "Buy the software licence yourself and we will refund you next week.",
    "The cheque was too large, so send back the extra money to my agent.",
    "Wire the overpaid balance back once the cheque clears.",
    "Deposit the registration amount today to hold your seat.",
    "Kindly pay Rs 2,500 for the starter kit.",
]
PAYMENT_NEGATIVE = [
    "We will never ask you to pay anything for this job.",
    "Do not pay any recruiter who asks for a fee.",
    "There is no fee to apply.",
    "A registration fee is not required.",
    "We will not charge you for training.",
    "Training is free of charge for all new joiners.",
    "No deposit is needed from you.",
    "Nothing to pay at any stage of hiring.",
    "We pay for your training and equipment ourselves.",
    "Your salary will be sent on the last working day.",
    "The company will transfer the joining bonus to your account.",
    "Why do I need to pay a fee first?",
    "I won't pay anything upfront.",
    "Let's discuss the role and the team.",
]
SENSITIVE_POSITIVE = [
    "Share your net banking login so we can set up payroll.",
    "Tell me your card number and the expiry date.",
    "What is the CVV on the back of the card?",
    "Read out the OTP that arrives on your phone.",
    "Send the SMS code you just received.",
    "I need your social security number for the background check.",
    "Please send a scan of your passport and a selfie.",
    "Upload a photo of your ID card.",
    "What was your mother's maiden name?",
    "Give me your bank credentials to verify the account.",
    "Reply with the verification code from the text message.",
    "Send the debit card details for activation.",
]
SENSITIVE_NEGATIVE = [
    "We will never ask for your OTP.",
    "Do not share your password with anyone, including us.",
    "Never share your card number or CVV with a recruiter.",
    "We do not need your bank login at any stage.",
    "No need to send your passport at this point.",
    "Your credentials are not required for the interview.",
    "We won't ask for your social security number over chat.",
    "Please don't send a selfie or ID photo to anyone online.",
    "The interview is on Monday at ten.",
    "Please confirm your availability for the call.",
    "Our team will share the job description shortly.",
    "Send me your resume when ready.",
]


@pytest.mark.parametrize("text", PAYMENT_POSITIVE)
def test_payment_positive(text):
    assert is_payment_request(text)


@pytest.mark.parametrize("text", PAYMENT_NEGATIVE)
def test_payment_negative(text):
    assert not is_payment_request(text)


@pytest.mark.parametrize("text", SENSITIVE_POSITIVE)
def test_sensitive_positive(text):
    assert asks_sensitive_information(text)


@pytest.mark.parametrize("text", SENSITIVE_NEGATIVE)
def test_sensitive_negative(text):
    assert not asks_sensitive_information(text)


def test_warning_sentence_does_not_hide_a_demand_in_the_same_message():
    assert is_payment_request("We never ask for money in interviews. Pay the processing fee now.")
