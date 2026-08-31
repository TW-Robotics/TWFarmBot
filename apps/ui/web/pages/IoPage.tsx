import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Selector } from "@astryxdesign/core/Selector";
import { Slider } from "@astryxdesign/core/Slider";
import { Switch } from "@astryxdesign/core/Switch";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, postAction } from "../api";
import { PageHeader } from "../components/PageHeader";

export function IoPage() {
  const toast = useToast();
  const [pins, setPins] = useState<any[]>([]);
  const [values, setValues] = useState<Record<number, unknown>>({});
  const [waterSecs, setWaterSecs] = useState(2);
  const [outputLabel, setOutputLabel] = useState("");
  const [analog, setAnalog] = useState(0);
  const [pulse, setPulse] = useState(true);
  const [pulseSecs, setPulseSecs] = useState(2);

  useEffect(() => {
    void api<{ pins?: any[] }>("/pins")
      .then((r) => {
        const list = r.pins || [];
        setPins(list);
        const first = list.find((p: any) => p.kind !== "sensor");
        if (first) setOutputLabel(`${first.label} · pin ${first.pin}`);
      })
      .catch(() => setPins([]));
  }, []);

  const sensors = pins.filter((p) => p.kind === "sensor");
  const outputs = pins.filter((p) => p.kind !== "sensor");
  const selected = outputs.find((p) => `${p.label} · pin ${p.pin}` === outputLabel);

  const writePin = async (pin: number, value: number, mode: string, seconds?: number) => {
    try {
      await postAction("write_pin", { pin, value, mode, ...(seconds ? { seconds } : {}) });
      toast({ body: `pin ${pin} = ${value}` });
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Pin write failed", type: "error" });
    }
  };

  return (
    <VStack gap={5}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="I/O workspace" />
      {sensors.length === 0 ? (
        <Text color="secondary">No sensor pins configured.</Text>
      ) : (
        <HStack gap={3}>
          {sensors.map((s) => (
            <Card key={s.pin} padding={3}>
              <VStack gap={2}>
                <Text weight="semibold">
                  {s.label} · {s.mode}
                </Text>
                <Text type="supporting">pin {s.pin}</Text>
                <Button
                  label="Read"
                  onClick={() =>
                    void api(`/pin/${s.pin}?mode=${s.mode || "analog"}`)
                      .then((r) => setValues((v) => ({ ...v, [s.pin]: r.value })))
                      .catch((error) => toast({ body: String(error.message), type: "error" }))
                  }
                />
                <Text>{String(values[s.pin] ?? "—")}</Text>
              </VStack>
            </Card>
          ))}
        </HStack>
      )}
      <HStack gap={3}>
        <Card padding={4}>
          <VStack gap={3}>
            <Text weight="semibold">Irrigation</Text>
            <NumberInput label="Seconds" value={waterSecs} onChange={setWaterSecs} min={0.1} max={300} step={0.5} />
            <Button
              label="Water"
              variant="primary"
              onClick={() =>
                void postAction("water", { seconds: waterSecs })
                  .then(() => toast({ body: "Queued" }))
                  .catch((error) => toast({ body: String(error.message), type: "error" }))
              }
            />
          </VStack>
        </Card>
        <Card padding={4}>
          <VStack gap={3}>
            <Text weight="semibold">Peripheral control</Text>
            {outputs.length === 0 ? (
              <Text color="secondary">No output pins configured.</Text>
            ) : (
              <>
                <Selector
                  label="Output"
                  options={outputs.map((p) => `${p.label} · pin ${p.pin}`)}
                  value={outputLabel}
                  onChange={setOutputLabel}
                />
                {selected?.mode === "analog" ? (
                  <>
                    <Slider label="PWM value" value={analog} onChange={setAnalog} min={0} max={255} />
                    <Button
                      label="Apply"
                      onClick={() => void writePin(selected.pin, analog, "analog")}
                    />
                  </>
                ) : selected ? (
                  <>
                    <Switch label="Timed pulse" isSelected={pulse} onChange={setPulse} />
                    {pulse && (
                      <NumberInput
                        label="Seconds"
                        value={pulseSecs}
                        onChange={setPulseSecs}
                        min={0.1}
                        max={300}
                        step={0.5}
                      />
                    )}
                    <HStack gap={2}>
                      <Button label="OFF" onClick={() => void writePin(selected.pin, 0, selected.mode || "digital")} />
                      <Button
                        label="ON"
                        variant="primary"
                        onClick={() =>
                          void writePin(
                            selected.pin,
                            1,
                            selected.mode || "digital",
                            pulse ? pulseSecs : undefined,
                          )
                        }
                      />
                    </HStack>
                  </>
                ) : null}
              </>
            )}
          </VStack>
        </Card>
      </HStack>
    </VStack>
  );
}
