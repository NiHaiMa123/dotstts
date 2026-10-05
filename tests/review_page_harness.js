"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const page = fs.readFileSync(process.argv[2], "utf8");
const match = page.match(/<script>\s*([\s\S]*?)<\/script>/);
assert(match, "review page script was not found");
const script = match[1].replace(/\nload\(\);\s*$/, "");

const alerts = [];
const downloads = [];
const storage = new Map();
const saveStatus = { textContent: "", className: "" };
const saveButton = { textContent: "保存本条" };
let blobIndex = 0;
const context = {
  Blob,
  console,
  alert(message) { alerts.push(String(message)); },
  setTimeout(callback) { callback(); return 1; },
  crypto: {
    counter: 0,
    randomUUID() {
      this.counter += 1;
      return `${String(this.counter).padStart(8, "0")}-0000-4000-8000-000000000000`;
    },
  },
  localStorage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); },
  },
  URL: {
    createObjectURL() { blobIndex += 1; return `blob:test-${blobIndex}`; },
    revokeObjectURL() {},
  },
  document: {
    body: { appendChild() {} },
    getElementById(id) {
      if (id === "save_status") return saveStatus;
      if (id === "save") return saveButton;
      return null;
    },
    createElement(tag) {
      if (tag === "a") {
        return {
          href: "",
          download: "",
          click() { downloads.push(this.download); },
        };
      }
      return {
        value: "",
        style: {},
        select() {},
        remove() {},
      };
    },
    execCommand() { return true; },
  },
};
vm.createContext(context);
vm.runInContext(script, context);

const asset = "a".repeat(64);
context.testSamples = [{ asset_sha256: asset }];
context.validDecision = {
  asset_sha256: asset,
  text_decision: "flag_defect",
  text_final: null,
  text_note: "待查",
  emotion_primary: "中立_neutral",
  emotion_secondary: null,
  intensity: null,
  review_status: "pending",
  auto_rules_applied: [],
};
vm.runInContext("samples = testSamples", context);
context.serializedDecision = JSON.stringify({ [asset]: context.validDecision });
const restored = vm.runInContext(
  "restoreDecisions(serializedDecision)", context
);
assert.equal(restored[asset].review_status, "pending");

context.unknownDecision = JSON.stringify({
  ["b".repeat(64)]: { ...context.validDecision, asset_sha256: "b".repeat(64) },
});
assert.throws(
  () => vm.runInContext("restoreDecisions(unknownDecision)", context),
  /非当前包资产/
);
assert.throws(
  () => vm.runInContext('restoreDecisions("{")', context),
  /JSON/
);

context.nextDecision = context.validDecision;
assert.equal(
  vm.runInContext(
    "collectDecision = () => nextDecision; decisions = {}; save()", context
  ),
  true
);
const storedDecisions = JSON.parse([...storage.values()][0]);
assert.equal(storedDecisions[asset].text_decision, "flag_defect");
assert(saveStatus.textContent.includes("已保存到本地草稿"));
assert(saveStatus.textContent.includes("pending"));

context.nextDecision = {
  ...context.validDecision,
  text_decision: "accept_edited",
  text_final: "",
  text_note: null,
  review_status: "approved",
};
assert.equal(vm.runInContext("save()", context), false);
assert(alerts.some((message) => message.includes("最终文本不能为空")));
assert(saveStatus.textContent.includes("未保存"));

context.nextDecision = context.validDecision;
vm.runInContext("decisions = { [nextDecision.asset_sha256]: nextDecision }", context);
vm.runInContext("save = () => true; exportDecisions(); exportDecisions()", context);
assert.equal(downloads.length, 2);
assert.notEqual(downloads[0], downloads[1]);

const activeKey = vm.runInContext("STORAGE_KEY", context);
storage.set(activeKey, "broken");
context.brokenDraft = "broken";
vm.runInContext(
  'quarantineStoredDraft(brokenDraft, new Error("bad draft"))', context
);
assert.equal(storage.has(activeKey), false);
assert([...storage.keys()].some((key) => key.startsWith(`${activeKey}-invalid-`)));

console.log("review page harness: ok");
