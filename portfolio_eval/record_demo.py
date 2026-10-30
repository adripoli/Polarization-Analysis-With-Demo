# record_demo py drives
import sys, os, time
from playwright.sync_api import sync_playwright

OUT = sys.argv[1]
VIDEO = "--video" in sys.argv
URL = os.environ.get("DEMO_URL", "http://127.0.0.1:7860/")
os.makedirs(OUT, exist_ok=True)


def pause(s):
    # only bother waiting
    time.sleep(s if VIDEO else 0.3)


def caption(page, text):
    # fixed bar at
    page.evaluate("""t => {
        let c = document.getElementById('pp-cap');
        if (!c) { c = document.createElement('div'); c.id = 'pp-cap';
            c.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:9999;background:rgba(11,11,11,.86);' +
              'color:#fff;font:600 19px/1.35 system-ui,sans-serif;padding:14px 28px;text-align:center';
            document.body.appendChild(c); document.body.style.paddingBottom = '110px'; }
        c.textContent = t; }""", text)


def wait_cards(page, n_before=0):
    page.wait_for_function(f"document.querySelectorAll('.pp-card').length > {n_before}", timeout=120000)


with sync_playwright() as p:
    b = p.chromium.launch()
    kw = dict(viewport={"width": 1280, "height": 860}, device_scale_factor=1)
    if VIDEO:
        kw.update(record_video_dir=OUT, record_video_size={"width": 1280, "height": 860})
    ctx = b.new_context(**kw)
    page = ctx.new_page()
    page.goto(URL); page.wait_for_selector("text=Polarization Pipeline: live demo")
    caption(page, "One tweet through every version I built. Marks 1-3 are the original keyword code from git.")
    pause(3.5)

    # tab 1
    box = page.get_by_label("Tweet")
    run = page.get_by_role("button", name="Run through every Mark")
    caps = ["The bug that started Mark 2: “MAGA” makes Mark 1 call this pro-Trump. Mark 2's flip fixes it. Mark 7 still misses this one.",
            "The regression: Mark 1 got this right, Mark 2 broke it, Mark 3's word-distance weighting fixed it again.",
            "Sarcasm: every rule-based version gives up. Mark 4 misreads it. Mark 7 gets it right.",
            "Why Mark 7 exists: Mark 4 learned “mentions Biden = Trump fan”. Mark 7 evened out who's mentioned, so it reads this right."]
    for i, t in enumerate(["Those MAGA people are crazy",
                           "Incompetent. Just like everyone else involved in the trump clown car. VoteBiden",
                           "Ugh, of course the MAGA crowd thinks this is brilliant. Great job guys. VoteBiden",
                           "I love Biden"]):
        page.evaluate("document.querySelectorAll('.pp-card').forEach(e => e.remove())")
        caption(page, caps[i])
        box.fill("")
        box.press_sequentially(t, delay=45 if VIDEO else 0)
        pause(0.6)
        page.evaluate("document.querySelectorAll('.pp-card').forEach(e => e.remove())")
        run.click(); wait_cards(page)
        page.locator('.pp-card').last.scroll_into_view_if_needed()
        pause(4.5)
        page.screenshot(path=os.path.join(OUT, f"shot_tab1_{i}.png"))

    # tab 2
    page.evaluate('window.scrollTo(0, 0)')
    caption(page, "Score a whole user with the final polarization formula, and see where they'd sit among 27,965 real users.")
    page.get_by_role("tab", name="2. Score a user").click(); pause(1.5)
    ubox = page.get_by_label("A user's tweets, one per line")
    ugo = page.get_by_role("button", name="Score this user")
    import json
    ex = json.load(open(r"C:\Users\Asus Gaming\Desktop\political_polarization\demo\data\example_users.json"))
    users = ["\n".join(ex["trump"]), "\n".join(["Get your official election 2020 shirts now at our store!"] * 4)]
    ucaps = ["A user made of 8 real pro-Trump test tweets. Mark 7 gets 7 of 8 right, so they land well right of center.",
             "A copy-paste bot: zero spread, so Mark 4's ranking called it the most polarized. Mark 5's dedupe removes it."]
    for i, u in enumerate(users):
        caption(page, ucaps[i])
        page.evaluate('window.scrollTo(0, 0)')
        ubox.fill(u); pause(0.8)
        ugo.click()
        # wait for THIS
        page.wait_for_selector("text=Every tweet is identical" if i == 1 else "text=Polarization score", timeout=120000)
        page.wait_for_timeout(1500)
        page.evaluate('window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"})')
        pause(5)
        page.screenshot(path=os.path.join(OUT, f"shot_tab2_{i}.png"), full_page=True)

    # tab 3
    page.evaluate('window.scrollTo(0, 0)')
    page.get_by_role("tab", name="3. Grade it yourself").click()
    caption(page, "Random held-out tweets: hand-labeled by me, never used in training. Watch the running tally.")
    pause(3)
    draw = page.get_by_role("button", name="Draw a random held-out tweet")
    for i in range(4):
        draw.click()
        page.wait_for_selector(f"text=/{i+1}\\/{i+1}|\\d\\/{i+1}/", timeout=120000)
        page.evaluate('window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"})')
        pause(3.5)
    page.screenshot(path=os.path.join(OUT, "shot_tab3.png"), full_page=True)
    pause(1.5)
    vid = page.video.path() if VIDEO else None
    ctx.close(); b.close()
    print("video:", vid)
