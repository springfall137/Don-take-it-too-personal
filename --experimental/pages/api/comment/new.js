import { connectDB } from "@/util/database";
import { ObjectId } from "mongodb";
import { getServerSession } from "next-auth";
import { authOptions } from "../auth/[...nextauth]";

export default async function handler(rq, rp) {
    let session = await getServerSession(rq, rp, authOptions)
    if (rq.method == 'POST') {
        rq.body = JSON.parse(rq.body)

        if (!session) {
            return rp.status(401).json('로그인 필요')
        } else {
            let save = {
                content: rq.body.comment,
                parent: new ObjectId(rq.body._id),
                author: session.user.email
            }

            const db = (await connectDB).db('forum')
            let result = await db.collection('comment').insertOne(save)
            rp.status(200).json('저장완료')
          }
    }
}
