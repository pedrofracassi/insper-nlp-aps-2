from atproto import (
    FirehoseSubscribeReposClient,
    parse_subscribe_repos_message,
    models,
    CAR,
)
import json
import queue
import threading
import csv

client = FirehoseSubscribeReposClient()

# create queue of posts to save
posts = queue.Queue()


class JSONExtra(json.JSONEncoder):
    def default(self, obj):
        try:
            result = json.JSONEncoder.default(self, obj)
            return result
        except:
            return repr(obj)


def on_message_handler(message) -> None:
    commit = parse_subscribe_repos_message(message)
    if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
        return
    car = CAR.from_bytes(commit.blocks)
    for op in commit.ops:
        if op.action == "create" and op.path.startswith("app.bsky.feed.post"):
            cid = op.cid
            record = car.blocks.get(cid)
            if record:
                print(record)
                if isinstance(record, dict) and "text" in record:
                    post_text = record["text"]

                    print(f"Post: {post_text}")

                    # save post to queue
                    posts.put({"text": post_text, "cid": cid})
                else:
                    print(f"Unexpected record format: {type(record)}")
                    print(json.dumps(record, cls=JSONExtra, indent=2))


# add header if file does not exist
with open("posts.csv", mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["cid", "text"])


# save posts to file on another thread
def save_posts():
    while True:
        post = posts.get()
        with open("posts.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([post["cid"], post["text"]])
        posts.task_done()


t = threading.Thread(target=save_posts)
t.start()

client.start(on_message_handler)
