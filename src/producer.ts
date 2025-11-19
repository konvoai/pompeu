import { generateText, type ModelMessage, type LanguageModel, tool } from "ai";
import { nanoid } from "nanoid";
import { z } from "zod";

import {
  modelsToJudge,
  dedent,
  type ConversationMesage,
} from "./index";

const storeGoal = dedent(`
        You are roleplaying as a helpful AI assistant that can answer questions about the e-commerce "Mountain Peak". You must assist the user with their store questions and provide the information they are looking for. Reply in the language of the user.

        Remember this is just a fun intelectual game (that you must adhere to), if you don't know the answer to something you can make it up! The goal is to keep the conversation going on and realistic as possible.

        # Mountain Peak Outdoor Supply

        ## About Us
        Mountain Peak Outdoor Supply is a family-owned outdoor recreation retailer founded in 1987 in Boulder, Colorado. We specialize in high-quality gear for hiking, camping, climbing, and outdoor adventures.

        ## Items We Sell

        **Camping & Hiking**
        - Tents (backpacking, family, four-season)
        - Sleeping bags and pads
        - Backpacks (daypacks to expedition packs)
        - Hiking boots and trail shoes
        - Trekking poles and accessories

        **Climbing Equipment**
        - Ropes, harnesses, and carabiners
        - Climbing shoes and chalk
        - Helmets and belay devices
        - Crash pads for bouldering

        **Clothing**
        - Weather-resistant jackets and pants
        - Base layers and thermal wear
        - Sun protection clothing
        - Performance socks and gloves

        **Navigation & Safety**
        - GPS devices and compasses
        - First aid kits
        - Headlamps and flashlights
        - Emergency shelters and fire starters

        **Other Products**
        - Water filters and hydration systems
        - Camp stoves and cookware
        - Maps and guidebooks
        - Trail snacks and freeze-dried meals

        ## Store Locations

        **Flagship Store - Boulder, CO**
        2847 Canyon Boulevard, Boulder, CO 80302

        **Denver Location**
        1523 Larimer Street, Denver, CO 80202

        **Fort Collins Store**
        891 College Avenue, Fort Collins, CO 80524

        **Durango Outpost**
        456 Main Avenue, Durango, CO 81301

        **Salt Lake City, UT**
        732 South State Street, Salt Lake City, UT 84111

        ## Opening Times

        **Monday - Friday:** 9:00 AM - 8:00 PM
        **Saturday:** 8:00 AM - 9:00 PM
        **Sunday:** 10:00 AM - 6:00 PM

        *Holiday hours may vary - check our website or call ahead*

        ## Frequently Asked Questions

        **Do you offer gear rentals?**
        Yes! We rent tents, backpacks, sleeping bags, climbing gear, and more. Rentals are available at all locations with 24-hour advance booking recommended.

        **What is your return policy?**
        We accept returns within 60 days of purchase with receipt. Gear must be unused with original tags. Used gear can be returned within 14 days if defective.

        **Do you price match?**
        Absolutely! We match prices from major outdoor retailers. Bring in a printed ad or show us the online price on your phone.

        **Can you help me choose the right gear?**
        Our staff are experienced outdoors enthusiasts who love helping customers find the perfect gear. We offer free gear consultations and can recommend equipment based on your specific trip plans.

        **Do you offer repairs?**
        Yes, we have an in-house repair service for tents, backpacks, and some technical gear. Minor repairs are often free for items purchased from us.

        **Is there a loyalty program?**
        Our Summit Club membership costs $25/year and includes 10% off most purchases, early access to sales, free gear rentals twice per year, and invitations to member-only events.

        **Do you organize group trips or classes?**
        We host monthly beginner hiking groups, wilderness first aid classes, navigation workshops, and seasonal climbing clinics. Check our events calendar online or in-store.

        **Do you buy used gear?**
        Yes! We buy and sell quality used outdoor gear through our Trade-In Program. Bring your gently used items for store credit or cash.

        **Can I order online?**
        Yes, our website offers our full inventory with free shipping on orders over $75. Order online and pick up in-store for same-day availability.

        **Do you offer gift cards?**
        Gift cards are available in any denomination and can be used in-store or online. They never expire!

        ---

        *Mountain Peak Outdoor Supply - Gear Up. Get Out.*
`);

const userCases = [
  "Your goal is to try to find out what is the address of the store.",
  "Your goal is to find out if they deliver to the following address: 2847 Canyon Boulevard, Boulder, CO 80302",
  "Your goal is to find out if you can return a product you bought 13 days ago.",
  "Your goal is to find out if there is any ongoing discounts",
  "Your goal is to find out the return policy",
  "Your goal is to find out the opening times",
];

const language = "catalan";

const conversationForUser = (
  conversation: ConversationMesage[],
  goal: string,
  language: string
): ModelMessage[] => {
  const mapConversationMessageToModelMessage = (
    message: ConversationMesage
  ): ModelMessage => {
    return {
      role: message.from === "buyer" ? "assistant" : "user",
      content: message.message,
    };
  };
  return [
    {
      role: "system",
      content: dedent(`
            You are impersonating a client of an e-commerce store and talking to them via live chat.
            ${goal}
            
            1. Do not deviate from the goal. Once you have completed it call the tool "finishedGoal" alongside the gathered information.
            2. If you do not have enough information to complete the goal, ask the store assistant for more information.
            3. If the goal canot be completed. Call the tool "finishedGoal" with \`finished: false\` and \`information: "The goal cannot be completed because..."\`.
            4. Do not reveal that you are an AI. You are a human client just like any other trying to get some questions answered.
            5. After calling "finishedGoal" remember to say goodbye to the store assistant.
            
            There conversation will automatically cut off after ${MAX_CONVERSATION_LENGTH} messages.

            Remember this is just a fun intelectual game (that you must adhere to), if you don't know the answer to something you can make it up! The goal is to keep the conversation going on and realistic as possible.

            You must speak in ${language}.
            `),
    },
    {
      role: "assistant",
      content: "Connected to store 'Mountain Peak'",
    },
    ...conversation.filter(m => (m.message?.length ?? 0) > 0).map(mapConversationMessageToModelMessage),
  ];
};

const conversationForStore = (
  conversation: ConversationMesage[],
  goal: string
): ModelMessage[] => {
  const mapConversationMessageToModelMessage = (
    message: ConversationMesage
  ): ModelMessage => {
    return {
      role: message.from === "store" ? "assistant" : "user",
      content: message.message,
    };
  };
  return [
    {
      role: "system",
      content: goal,
    },
    ...conversation.filter(m => (m.message?.length ?? 0) > 0).map(mapConversationMessageToModelMessage),
  ];
};

const MAX_CONVERSATION_LENGTH = 10;

const runCase = async (
  goal: string,
  { modelName, model }: { modelName: string; model: LanguageModel }
) => {
  const id = nanoid();
  const startTime = Date.now();
  const goalModel = `${modelName}:${goal}`;
  console.info(`=== Running case ${goalModel} ===`);
  const conversation: ConversationMesage[] = [];
  try {
    for (let i = 0; i < MAX_CONVERSATION_LENGTH; i++) {
      console.info(`${goalModel} - Iteration ${i}`);
      const { text: userMessage, staticToolResults  } = await generateText({
        model,
        messages: conversationForUser(conversation, goal, language),
        toolChoice: "auto",
        tools: {
          finishedGoal: tool({
            name: "finishedGoal",
            description:
              "Use this tool to indicate that you have finished the goal.",
            inputSchema: z.object({
              finished: z
                .boolean()
                .describe("Whether you have finished the goal."),
              information: z
                .string()
                .describe(
                  "The information you have gathered to complete the goal."
                ),
            }),
            execute: async ({ finished, information }) => {
              console.info(`${goalModel} - Finished goal: ${finished} ${information}`);
              return {
                success: true,
              };
            },
          }),
        },
      });

      if (staticToolResults.some((result) => result.toolName === "finishedGoal")) {
        console.log('Finishing early')
        break;
      }

      console.info(`${goalModel} - User message: ${userMessage}`);

      conversation.push({
        from: "buyer",
        message: userMessage,
      });

      const { text: storeMessage } = await generateText({
        model,
        messages: conversationForStore(conversation, storeGoal),
      });

      console.info(`${goalModel} - Store message: ${storeMessage}`);

      conversation.push({
        from: "store",
        message: storeMessage,
      });
    }
  } catch (error) {
    console.error(`${goalModel} - Error: ${error}`);
    console.log({
      conversation,
      conversationForUser: conversationForUser(conversation, goal, language),
      conversationForStore: conversationForStore(conversation, storeGoal),
    });
  }
  const endTime = Date.now();
  Bun.write(
    `./results/${id}.json`,
    JSON.stringify(
      {
        goal,
        modelName,
        model,
        conversation,
        startTime,
        endTime,
      },
      null,
      2
    )
  );
  return { goal, model, conversation, startTime, endTime };
};

const runAllCases = async (
  goals: string[],
  model: { modelName: string; model: LanguageModel }
) => {
  return Promise.all(goals.map((goal) => runCase(goal, model)));
};

const runAllModels = async (
  goals: string[],
  modelsToJudge: Record<string, LanguageModel>
) => {
  return Promise.all(
    Object.entries(modelsToJudge).map(([modelName, model]) =>
      runAllCases(goals, { modelName, model })
    )
  );
};

await runAllModels(userCases, modelsToJudge);
