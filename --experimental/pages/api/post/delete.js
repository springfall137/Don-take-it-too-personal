import { ObjectId } from "mongodb"
import { connectDB } from "@/util/database"
import { getServerSession } from "next-auth"
import { authOptions } from "../auth/[...nextauth]"

export default async function handler(request, response) {
    if (request.method == 'POST') {
        console.log(request.body)
        let session = await getServerSession(request, response, authOptions)

        const db = (await connectDB).db('forum')
        let author = await db.collection('post').findOne({ _id: new ObjectId(request.body) })

        if (author.author == session.user.email) {
            let result = await db.collection('post').deleteOne({ _id: new ObjectId(request.body) })
            return response.status(200).json('삭제완료')
        } else {
            return response.status(500).json('현재유저와 작성자 불일치')
        }
    }
}
