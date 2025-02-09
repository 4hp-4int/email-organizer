"""
created this manually by reviewing the topic information from the emailClassifer-50 model
created top level labels, then went through and grouped topics by their label.
"""

topic_labels_llama = {
    -1: "Tech Deals",
    0: "Shopping",
    1: "Clothing Sale",
    2: "Financial Transactions",
    3: "Professional Networking",
    4: "Administrative",
    5: "Games Wishlist",
    6: "Tech Deals",
    7: "Music Events in St. Louis",
    8: "Real Estate Investing",
    9: "Credit Reports",
    10: "NBA",
    11: "Daily Agendas",
    12: "Eyewear Sale",
    13: "Coffee",
    14: "Marijuana Reform",
    15: "Handyman Services",
    16: "Server Alerts",
    17: "Subscription Renewal Reminders",
    18: "Real Estate",
    19: "Hair Care",
    20: "BART Board Updates",
    22: "Pokemon Cards",
    23: "Fitness",
    24: "Health and Wellness Program",
    26: "Skincare",
    27: "Software Development",
    28: "Work Reports",
    29: "Student Loan Information",
    30: "WB Games",
    31: "Shopping AI",
    32: "Zimbabwe Archaeology",
    32: "Bevel Electric Shave Essentials Kit",
    34: "NordVPN",
    35: "Air Travel",
}

topic_labels = {
    -1: "unknown",
    0: "Online Shopping",
    1: "Orders",
    2: "Technology",
    3: "Bills, Statements and Receipts",
    4: "Entertainment - PC Gaming",
    5: "Retail - Clothing and Apparel",
    6: "Entertainment - Media",
    7: "Promotions - PC Gaming",
    8: "Entertainment - Media",
    9: "Administrative",
    10: "Bills, Statements and Receipts",
    11: "Retail - Clothing and Apparel",
    12: "Finances",
    13: "Administrative",
    14: "Entertainment- Sports",
    15: "Finances",
    16: "Real Estate",
    17: "Coffee",
    18: "ISP",
    19: "Politics - Marijuana",
    20: "Promotions",
    21: "Bills, Statements and Receipts",
    22: "Healthcare",
    23: "Finances",
    24: "Hosted Services",
    25: "Pokemon",
    26: "Politics",
    27: "Software Development",
    28: "Promotions",
    29: "Promotions",
    30: "Hosted Services",
    31: "Administrative",
    32: "Bills, Statements and Receipts",
    33: "Entertainment - Media",
}


topic_labels_13 = {
    -1: "Offers",
    0: "Orders",
    1: "Career - Technology",
    2: "Administrative",
    3: "Agenda",
    4: "Local - St. Louis",
    5: "Entertainment",
    6: "Finances",
    7: "Politics",
    8: "Healthcare",
    9: "Transit",
    10: "Development",
    11: "Miscellaneous",
}

topic_labels_50 = {
    -1: "Promotions",
    0: "Offers",
    1: "Career, Technology",
    2: "Orders",
    3: "Promotions",
    4: "Promotions",
    5: "Promotions",
    6: "Entertainment",
    7: "Bills",
    8: "Promotions",
    9: "PC Gaming",
    10: "Entertainment",
    11: "Finances",
    12: "Promotions",
    13: "Account Access",
    14: "Finances",
    15: "Entertainment",
    16: "Agenda",
    17: "Real Estate",
    18: "Finances",
    19: "Coffee",
    20: "PC Gaming",
    22: "Promotions",
    23: "Account Access",
    24: "Promotions",
    26: "Finances",
    27: "Bills",
    31: "Entertainment",
    34: "Bills",
    35: "Bills",
    36: "Receipts",
    37: "Promotions",
    38: "Promotions",
    39: "Promotions",
    44: "Promotions",
    45: "Bills",
    46: "Entertainment",
    47: "Bills",
    48: "Account Access",
}

"""
Subject: Khalen, gain a competitive edge at these upcoming events for Premium subscribers - Label: Career, Technology
Subject: Get Valentine’s Day gifts with flexible payment plans - Label: Offers/Reviews
Subject: Ready for Super Bowl Sunday? - Label: Social Media
Subject: Local store deals are waiting for you 👀 - Label: Offers/Reviews
Subject: BART Board Updates - Label: Politics
Subject: Zero sum games | DEV Digest - Label: Career, Technology
Subject: Spice up your Dungeons & Dragons game night with awesome new campaigns - Label: Offers/Promotions
Subject: Let’s make the most of your retirement plan with Vanguard advice - Label: Finances
Subject: Restock Alert: Humanrace x Dover Street Market - Label: Bills
Subject: NBA trade deadline buzz: Latest updates | All-Star Game uniforms and court design unveiled - Label: Entertainment
Subject: AFTER HOURS TIL DAWN TOUR: GET PRESALE TICKETS NOW! 🎤 - Label: Entertainment
Subject: The Humble Store Winter Sale is almost over! ⏰ - Label: Video Games/PC Gaming
Subject: Khalen, get up to $600 when you invest for retirement with Merrill - Label: Finances
Subject: We can see your future 🔮✨ - Label: Promotions
Subject: Launch Your Daycare Website Today with Our New Template! - Label: Offers/Promotions
Subject: Versatile insulated jackets—packable to parka - Label: Offers/Reviews
Subject: We Want to Hear from You! 📋 - Label: Promotions
Subject: Your receipt from Apple. - Label: Bills
Subject: Bespoke AI Appliances—up to $2,500 off Presidents' Day deals - Label: Promotions
Subject: Your January financial report - Label: Finances
Subject: GorillaT at Old Rock House and Loud Luxury at Ryse Nightclub This Friday!🕺 - Label: Entertainment
Subject: A shipment from order #KEND20338 is on the way - Label: Bills
Subject: 💌all our Sourdough love! 💌 - Label: Offers/Reviews
Subject: Big News: Bourbon & Beyond 2025 Lineup Is Out Now - Label: Entertainment
Subject: Good advice: Try Digital Advisor with no service fees - Label: Finances
Subject: Upgrade Your PC for Elite Level Frame Rates and Ultra-fast Renders - Label: Promotions
Subject: 🏈👓 NEW Drop: Von Miller x GlassesUSA.com ➡️ Early access offer inside - Label: Promotions
Subject: Save $50 Off A Deep Clean By Booking Today! - Label: Promotions
Subject: Don’t Miss This: Off-Market Property in St. Louis for Only $105K - Label: Real Estate
Subject: What Okta Bcrypt incident can teach us about designing better APIs | Mykola in ITNEXT - Label: Career, Technology
Subject: Rain? Check! Lightweight layers for whatever the weather - Label: Promotions
Subject: You have no events scheduled today. - Label: Calendar/Agenda
Subject: We’re updating our Subscriber Agreement - Label: Offers/Promotions
Subject: A shipment from order #KEND20338 is on the way - Label: Bills
Subject: We're making some changes to our PayPal legal agreements - Label: Bills
Subject: 2/5 CTDS - All Your Base Are Belong To Elon - Label: Social Media
Subject: Your Purchases Return Credits - Label: Promotions
Subject: To pair with what you purchased - Label: Offers/Reviews
Subject: Pay ONE CENT for 3 Gifts! - Label: Promotions
Subject: Khalen, pay bills easier - Label: Finances
Subject: Your ride with Shamsul on February 5 - Label: Offers/Reviews

"""
